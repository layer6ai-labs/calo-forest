import h5py
from pathlib import Path
import numpy as np


def get_raw_data_arrays(cfg: dict, name: str, root: str, split: str):
    # Train and test filenames.
    filenames = {
        "photons1": [['dataset_1_photons_1.hdf5'], ['dataset_1_photons_2.hdf5']],
        "pions1": [['dataset_1_pions_1.hdf5'], ['dataset_1_pions_2.hdf5']],
        }

    data_path = lambda x: Path(root) / x

    showers = []
    energies = []

    if split == "train":
        datasets = filenames[name][0]
    elif split == "test":
        datasets = filenames[name][1]
    for dataset in datasets:
        with h5py.File(data_path(dataset), "r") as h5f:
            if cfg["dataset_samples"] > 0:
                energy = h5f['incident_energies'][:cfg["dataset_samples"]].astype(np.float32)
                shower = h5f['showers'][:cfg["dataset_samples"]].astype(np.float32)
            else:
                energy = h5f['incident_energies'][:].astype(np.float32)
                shower = h5f['showers'][:].astype(np.float32)

            showers.append(shower)
            energies.append(energy)
    if showers: # if empty skip this
        shape = list(showers[0].shape)
        shape[0] = -1
        showers = np.reshape(showers, shape)
        energy_shape = list(energies[0].shape)
        energy_shape[0] = -1
        energies = np.reshape(energies, energy_shape)

    return showers, energies


def get_physics_datasets(cfg, name, data_root, valid_fraction):
    if not name in ["photons1", "pions1"]:
        raise ValueError(f"Unknown dataset {name}")

    train_showers, train_energies = get_raw_data_arrays(cfg, name, data_root, "train")
    test_showers, test_energies = get_raw_data_arrays(cfg, name, data_root, "test")

    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(train_showers.shape[0])
    shuffled_showers = train_showers[perm]
    shuffled_energies = train_energies[perm]

    # Currently hardcoded; could make configurable
    if valid_fraction >= 1.0:
        raise ValueError(f"valid_fraction cannot be greater than 1. Was {valid_fraction}")
    elif valid_fraction < 0.0:
        valid_fraction = 0.0
    valid_size = int(valid_fraction * train_showers.shape[0])
    valid_showers = shuffled_showers[:valid_size]
    valid_energies = shuffled_energies[:valid_size]
    train_showers = shuffled_showers[valid_size:]
    train_energies = shuffled_energies[valid_size:]

    dataset_dict = {}
    dataset_dict["train"] = [train_showers, train_energies.squeeze()]
    dataset_dict["valid"] = [valid_showers, valid_energies.squeeze()]
    dataset_dict["test"] = [test_showers, test_energies.squeeze()]
    dataset_dict["cat_indexes"] = [[], True] # all X data is real valued, not cat, bin, or int. y label is cat.
    dataset_dict["bin_indexes"] = [[], None]
    dataset_dict["int_indexes"] = [[], None]

    return dataset_dict
