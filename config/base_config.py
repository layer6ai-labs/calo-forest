def get_base_config(dataset):
    base_info = {
        "seed": 0,
        "ngen": 5,
        "log_delay": -1.0, # delay in seconds between logging system stats
        "data_root": "data/",
        "logdir_root": "runs/"
    }

    dataset_info = {
        "dataset_name": dataset,
        "valid_fraction": 0.0, # Fraction of train dataset to use for validation
        "dataset_samples": -1, # If integer > 0, use only that many samples for training
    }
    if dataset == 'random':
        dataset_info['num_random_datapoints'] = 1000
        dataset_info['num_random_features'] = 10
        dataset_info['num_random_classes'] = 10

    forest_diffusion_info = {
        "diffusion_type": 'vp',
        "n_t": 50,
        "duplicate_K": 100,
        "ycond": True,
        "eps": 0.001,
        "beta_min": 0.1,
        "beta_max": 8,
        "solver": "euler",
        "scaler": "min_max",
        "multinomial": False,
        "n_jobs": -1,
        "backend": "loky", # "threading", "loky" for joblib
        "n_batch": -1,
    }

    xgb_info = {
        "xgb_n_jobs": -1,
        "max_depth": 7,
        "n_estimators": 100,
        "eta": 0.3,
        "min_child_weight": 1,
        "gamma": 0,
        "lambda": 0,
        "multi_strategy": "one_output_per_tree", # either one_output_per_tree or multi_output_tree
        "early_stopping_rounds": None,
        "device": "cpu",
    }

    return {
        **base_info,
        **dataset_info,
        **forest_diffusion_info,
        **xgb_info,
    }
