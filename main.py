import os
import time
import argparse
import traceback
import pprint
import torch
import numpy as np
import pandas as pd

from system_logger import SystemLogger
from config import get_config, parse_config_arg, load_config_from_run_dir
from caloforest import get_writer, Writer
from caloforest.datasets import get_loaders_from_config
from caloforest.evaluators import evaluate_physics_samples, evaluate_tabular_samples
from caloforest.forest_diffusion import ForestModel


parser = argparse.ArgumentParser(description="Scaled up implementation of ForestDiffusion and ForestFlow")

parser.add_argument("--dataset", type=str,
    help="Dataset to train on. Required if resume_dir not specified.")
parser.add_argument("--config", default=[], action="append",
    help="Override config entries. Specify as `key=value`.")
parser.add_argument("--resume-dir", type=str,
    help="Directory to resume training from.")
parser.add_argument("--skip-eval", action="store_true",
    help="Skip evaluation of generated data.")

args = parser.parse_args()

if args.resume_dir is None:
    cfg = get_config(dataset=args.dataset)
    cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}
    writer = get_writer(args, cfg=cfg)
else:
    cfg = load_config_from_run_dir(run_dir=args.resume_dir)
    args.dataset = cfg['dataset']
    writer = Writer(
        logdir=args.resume_dir, # use the same directory for writing
        make_subdir=False,
        tag_group=args.dataset,
    )

pprint.sorted = lambda x, key=None: x
pp = pprint.PrettyPrinter(indent=4)
print(10*"-" + "cfg" + 10*"-")
pp.pprint(cfg)

np.random.seed(seed=cfg["seed"])
torch.manual_seed(cfg["seed"]) # torch is only used for calorimeter evaluation
torch.cuda.manual_seed(cfg["seed"])

if cfg["log_delay"] > 0.0:
    delay = cfg["log_delay"]
    print(f"Logging system stats every {delay}s.")
    print("sudo access is required to clear cache memory for accurate resource benchmarking.")
    os.system("sync")
    os.system("sudo sh -c 'echo 3 >/proc/sys/vm/drop_caches'")
    logger = SystemLogger(writer.logdir, delay=delay)
    logger.start()

try:
    dataset_dict = get_loaders_from_config(
        cfg,
        cfg["dataset"],
        cfg.get("data_root", "data/"),
        cfg["valid_fraction"]
    )

    # Transform dataset into format expected by ForestDiffusion
    X = dataset_dict["train"][0]
    y = dataset_dict["train"][1]
    if cfg["ycond"] and (dataset_dict["cat_indexes"][1] or dataset_dict["bin_indexes"][1]):
        y_uni = np.unique(y)
        print(f"There are {y_uni.shape[0]} classes for dataset {cfg['dataset']}.")
        print(f"X shape is {X.shape}")
        y_uni = np.sort(y_uni)
        label_map = {val.item(): idx for idx, val in enumerate(y_uni)}
        writer.write_json("label_map", label_map)
        map_func = np.vectorize(lambda x: label_map[x])
        y = map_func(y)
    else:
        X = np.concatenate([X, y[:, np.newaxis]], axis=1)
        print(f"Not using y_cond")
        print(f"Merging X and y, joint shape is {X.shape}")
        y = None
    
    # Package XGB hyperparameters
    hyper_names = ["max_depth", "n_estimators", "eta", "min_child_weight", "gamma", 
                   "lambda", "multi_strategy", "early_stopping_rounds", "device"]
    xgb_hypers = {k: v for k, v in cfg.items() if k in hyper_names}
    xgb_hypers["n_jobs"] = cfg["xgb_n_jobs"]

    if args.resume_dir is None:
        print("Starting forest_model")
        forest_model = ForestModel(
            n_t=cfg["n_t"],
            diffusion_type=cfg["diffusion_type"],
            xgb_hypers=xgb_hypers,
            duplicate_K=cfg["duplicate_K"],
            cat_indexes=dataset_dict["cat_indexes"][0],
            bin_indexes=dataset_dict["bin_indexes"][0],
            int_indexes=dataset_dict["int_indexes"][0],
            eps=cfg["eps"],
            beta_min=cfg["beta_min"],
            beta_max=cfg["beta_max"],
            solver=cfg["solver"],
            scaler=cfg["scaler"],
            n_jobs=cfg["n_jobs"],
            backend=cfg["backend"],
            n_batch=cfg["n_batch"],
            seed=cfg["seed"],
            logdir=writer.logdir,
        )
    else:
        print("Loading forest_model")
        forest_model = writer.load_pickle('forest_model')
    
    prepro_X = forest_model.preprocess(
        X=X,
        label_y=y # can be None
    )
    # Save model wrapper as pickle
    writer.write_pickle('forest_model', forest_model)

    t0 = time.time()
    forest_model.train(prepro_X)
    t1 = time.time()
    print(f"Done forest_model training in {t1-t0}s")
    if cfg["log_delay"] > 0.0 and logger.on():
        print("Closing system logger")
        logger.finish()
        max_mem, min_mem = logger.plot_system_usage()
    
    # We load the models into memory all at once so that generation timing
    # only count actual model usage. However, this causes larger peak
    # memory burden during generation. Models could be loaded on-the-fly
    # for each time step of generation if only doing a single batch.
    forest_model.load_models()

    if cfg["dataset"] in ["photons1", "pions1"]:
        # generate one set of samples using the labels from the train set
        t2 = time.time()
        Xy_gen = forest_model.generate(batch_size=X.shape[0], label_y=y)
        t3 = time.time()
        print(f"Generated data in {t3-t2}s")

        # Data transformations to match hdf5 file format
        X_gen = Xy_gen[:, :-1]
        y_gen = Xy_gen[:, -1]
        if cfg["ycond"]:
            undo_label_map = {val:key for key, val in label_map.items()}
            undo_map_func = np.vectorize(lambda x: undo_label_map[x])
            y_gen = undo_map_func(y_gen)
        y_gen = y_gen[..., np.newaxis]

        data_dict = {
            'incident_energies': y_gen,
            'showers': X_gen,
        }
        writer.write_hdf5('generated_showers', data_dict)

        if not args.skip_eval:
            evaluate_physics_samples(X_gen, y_gen, cfg, writer.logdir)
    else:
        # Tabular UCI datasets
        # generate ngen sets of samples using the distribution of labels from the train set
        ngen = cfg["ngen"]
        t2 = time.time()
        if cfg["ycond"] and not cfg["multinomial"] and (dataset_dict["cat_indexes"][1] or dataset_dict["bin_indexes"][1]):
            print("Using y_train labels for conditional generation")
            label_y = np.tile(y, ngen)
            Xy_gen = forest_model.generate(batch_size=ngen*X.shape[0], label_y=label_y)
        else:
            print("Doing unconditional generation using multinomial distribution over labels")
            Xy_gen = forest_model.generate(batch_size=ngen*X.shape[0], label_y=None)
        Xy_gen = np.split(Xy_gen, ngen)
        t3 = time.time()
        print(f"Generated {ngen} sets of data of size {X.shape[0]} in {t3-t2}s")
        timing_text = f"{t1-t0}\n{t3-t2}"
        if cfg["log_delay"] > 0.0 and not logger.on():
            timing_text = timing_text + f"\n{max_mem}\n{min_mem}"
        writer.write_textfile('timing', timing_text)

        if not args.skip_eval:
            metrics = pd.DataFrame()
            for gen_i in range(ngen):
                Xy_gen_i = Xy_gen[gen_i]
                X_gen = Xy_gen_i[:, :-1]
                y_gen = Xy_gen_i[:, -1]
                if cfg["ycond"] and (dataset_dict["cat_indexes"][1] or dataset_dict["bin_indexes"][1]):
                    undo_label_map = {val:key for key, val in label_map.items()}
                    undo_map_func = np.vectorize(lambda x: undo_label_map[x])
                    y_gen = undo_map_func(y_gen)
                metrics_dict = evaluate_tabular_samples(X_gen, y_gen, dataset_dict, cfg)
                print(metrics_dict)
                if gen_i == 0:
                    metrics_df = pd.DataFrame(metrics_dict, index=[0])
                else:
                    row_df = pd.DataFrame(metrics_dict, index=[0])
                    metrics_df = pd.concat([metrics_df, row_df], ignore_index=True)
            writer.write_pandas('metrics', metrics_df)
            mean_df = metrics_df.mean()
            stderr_df = metrics_df.sem()
            for i in range(metrics_df.shape[1]):
                print(f"{metrics_df.columns[i]}: {mean_df[i]:.3f} ± {stderr_df[i]:.3f}")

except Exception:
    traceback.print_exc()

finally:
    if cfg["log_delay"] > 0.0 and logger.on():
        print("Closing system logger")
        logger.finish()
        logger.plot_system_usage()
