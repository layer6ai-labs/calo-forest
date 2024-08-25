#!/usr/bin/env python3

import argparse
import time
import random
import numpy as np
import pandas as pd

from caloforest import get_writer
from config import load_config_from_run_dir
from caloforest.datasets import get_loaders_from_config
from caloforest.evaluators import evaluate_tabular_samples

t0 = time.time()

parser = argparse.ArgumentParser(description="Script for Sampling from ForestDiffusion model")

parser.add_argument("--load-dir", type=str, default="",
    help="Directory to load from.")
parser.add_argument("--evaluate", action="store_true",
    help="Evaluate showers after sampling, takes extra time.")
parser.add_argument("--solver", type=str, default=None)
parser.add_argument("--multinomial", action="store_true",
    help="Use multinomial sampling over y_train distribution rather than y_train directly.")
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()

cfg = load_config_from_run_dir(run_dir=args.load_dir)
args.dataset = cfg['dataset']
if args.solver is not None:
    cfg["solver"] = args.solver
cfg["multinomial"] = args.multinomial
if args.seed is not None:
    cfg["seed"] = args.seed # overwrite seed used in training
writer = get_writer(args, write_cfg=True, cfg=cfg)

# Set random seeds for reproducibility
np.random.seed(seed=cfg["seed"])
random.seed(cfg["seed"])

dataset_dict = get_loaders_from_config(
    cfg,
    cfg["dataset"],
    cfg.get("data_root", "data/"),
    cfg["valid_fraction"]
)
X = dataset_dict["train"][0]
y = dataset_dict["train"][1]

if cfg["ycond"] and (dataset_dict["cat_indexes"][1] or dataset_dict["bin_indexes"][1]):
    label_map = writer.load_json("label_map", args.load_dir)
    label_map = {float(k): v for k, v in label_map.items()}
    map_func = np.vectorize(lambda x: label_map[x])
    y = map_func(y)

# Sampling routine
forest_model_loaded = writer.load_pickle('forest_model', args.load_dir)
forest_model_loaded.set_logdir(args.load_dir)
forest_model_loaded.set_solver_fn(cfg["solver"])
print(f"Generating samples with the {cfg['solver']} solver")

ngen = cfg["ngen"]

if not args.multinomial and cfg["ycond"] and (dataset_dict["cat_indexes"][1] or dataset_dict["bin_indexes"][1]):
    print("Using y_train labels for conditional generation")
    multinom = False
else:
    if args.multinomial and cfg["ycond"] and (dataset_dict["cat_indexes"][1] or dataset_dict["bin_indexes"][1]):
        print("Using multinomial sampling of y labels for conditional generation")
    else:
        print("Model not conditional on y, doing unconditional generation")
    multinom = True

t2 = time.time()
if not multinom:
    label_y = np.tile(y, ngen)
    Xy_fake = forest_model_loaded.generate(batch_size=ngen*X.shape[0], label_y=label_y, seed=args.seed)
else:
    Xy_fake = forest_model_loaded.generate(batch_size=ngen*X.shape[0], label_y=None, seed=args.seed)
Xy_gen = np.split(Xy_fake, ngen)
t3 = time.time()

print(f"Generated {ngen} sets of data of size {X.shape[0]} in {t3-t2}s")
timing_text = f"0\n{t3-t2}"
timing_text = timing_text + "\n0\n0"
writer.write_textfile('timing', timing_text)

if args.evaluate:
    metrics = pd.DataFrame()
    for gen_i in range(ngen):
        Xy_fake_i = Xy_gen[gen_i]
        X_gen = Xy_fake_i[:, :-1]
        y_gen = Xy_fake_i[:, -1]
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
