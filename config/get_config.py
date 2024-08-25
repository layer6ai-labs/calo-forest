import os
import json

from .base_config import get_base_config
_VALID_DATASETS = ['photons1', 'pions1',
                   'iris', 'wine', 'california', 'parkinsons', \
                   'climate_model_crashes', 'concrete_compression', \
                   'yacht_hydrodynamics', 'airfoil_self_noise', \
                   'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
                   'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', \
                   'blood_transfusion', 'breast_cancer_diagnostic', \
                   'connectionist_bench_vowel', 'concrete_slump', \
                   'wine_quality_red', 'wine_quality_white', 'bean',
                   'tictactoe', 'congress', 'car', 'higgs', 'random']

def get_config(dataset):
    assert dataset in _VALID_DATASETS, \
        f"Unknown dataset {dataset}"

    base_config = get_base_config(dataset)

    return {
        **base_config,
        "dataset": dataset,
    }

def load_config_from_run_dir(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    return cfg

def load_config(args):
    cfg = load_config_from_run_dir(args.load_dir)

    if args.max_epochs_loaded:
        cfg["max_epochs"] = args.max_epochs_loaded

    return cfg
