#!/usr/bin/env python3

import argparse
import time
import h5py
import random
import numpy as np
from pathlib import Path

from caloforest import get_writer
from config import load_config_from_run_dir
from caloforest.evaluators import evaluate_physics_samples

t0 = time.time()

parser = argparse.ArgumentParser(description="Script for Sampling from ForestDiffusion model")

parser.add_argument("--load-dir", type=str, default="",
    help="Directory to load from.")
parser.add_argument("--batch-size", type=int, default=-1,
    help="Batch size for sampling.")
parser.add_argument("--evaluate", action="store_true",
    help="Evaluate showers after sampling, takes extra time.")
parser.add_argument("--solver", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()

 # Set random seeds for reproducibility
np.random.seed(seed=args.seed)
random.seed(args.seed)

cfg = load_config_from_run_dir(run_dir=args.load_dir)
args.dataset = cfg['dataset']
if args.solver is not None:
    cfg["solver"] = args.solver
if args.seed is not None:
    cfg["seed"] = args.seed # overwrite seed used in training
writer = get_writer(args, write_cfg=True, cfg=cfg)

# Set random seeds for reproducibility
np.random.seed(seed=cfg["seed"])
random.seed(cfg["seed"])

filenames = {
        "photons1": 'dataset_1_photons_2.hdf5',
        "pions1": 'dataset_1_pions_2.hdf5',
        }

data_path = lambda x: Path(cfg.get("data_root", "data/")) / x

dataset = filenames[cfg['dataset']]
with h5py.File(data_path(dataset), "r") as h5f:
    y = h5f['incident_energies'][:].astype(np.float32)

# Transform dataset into format expected by ForestDiffusion
label_map = writer.load_json("label_map", args.load_dir)
y_uni = np.unique(y)
y_uni = np.sort(y_uni)
label_map = {val: np.int32(idx) for idx, val in enumerate(y_uni)}
map_func = np.vectorize(lambda x: label_map[x])
y = map_func(y)
y = y.squeeze()
print("Dataset size")
print(y.shape)

# Sampling routine
forest_model_loaded = writer.load_pickle('forest_model', args.load_dir)
forest_model_loaded.set_logdir(args.load_dir)
forest_model_loaded.set_solver_fn(cfg["solver"])
print(f"Generating samples with the {cfg['solver']} solver")

if args.batch_size < 0: # single batch the size of y
    print(f"Sampling with batch size {y.shape[0]}")
    tt0 = time.time()
    Xy_gen = forest_model_loaded.generate(batch_size=y.shape[0], label_y=y, seed=args.seed)
    tt1 = time.time()
    print(f"{Xy_gen.shape[0]} conditional samples generated in {tt1 - tt0} seconds")
else: # generate in batches
    # sort y to reduce the number of predict calls in each batch
    y_sort = np.sort(y)

    num_generated = 0
    batch_size = args.batch_size
    print(f"Sampling with batch size {batch_size}")
    Xy_gen = []
    while num_generated < y_sort.shape[0]:
        tt0 = time.time()
        if num_generated + batch_size > y_sort.shape[0]:
            y_batch = y_sort[num_generated:] # final batch of smaller size
            batch_size = y_batch.shape[0]
        else:
            y_batch = y_sort[num_generated:num_generated+batch_size]

        Xy_batch = forest_model_loaded.generate(batch_size=batch_size, label_y=y_batch, seed=args.seed)
        Xy_gen.append(Xy_batch)
        num_generated += batch_size
        tt1 = time.time()
        print(f"{Xy_batch.shape[0]} conditional samples generated in {tt1 - tt0} seconds")
    # Concatenate batches into single array
    Xy_gen = np.concatenate(Xy_gen, axis=0)
# Data transformations to match hdf5 file format
print("Final generated shape:", Xy_gen.shape)
print("Final generated dtype:", Xy_gen.dtype)
X_gen = Xy_gen[:, :-1]
y_gen = Xy_gen[:, -1]
undo_label_map = {val:key for key, val in label_map.items()}
undo_map_func = np.vectorize(lambda x: undo_label_map[x])
y_gen = undo_map_func(y_gen)
y_gen = y_gen[..., np.newaxis]
data_dict = {
    'incident_energies': y_gen,
    'showers': X_gen,
}
writer.write_hdf5('generated_showers', data_dict)

if args.evaluate:
    evaluate_physics_samples(X_gen, y_gen, cfg, writer.logdir)

t1 = time.time()
print(f"CaloForest sampling script total time: {t1 - t0} seconds")
