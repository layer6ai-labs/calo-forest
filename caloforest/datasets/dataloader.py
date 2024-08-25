from .calo_challenge_showers import get_physics_datasets
from .tabular_datasets import get_tabular_datasets


def get_loaders_from_config(cfg,
        dataset,
        data_root,
        valid_fraction
        ):
    
    if dataset in ['photons1', 'pions1']:
        dataset_dict = get_physics_datasets(cfg,
            dataset,
            data_root,
            valid_fraction
            )
    else:
        dataset_dict = get_tabular_datasets(cfg,
            dataset,
            data_root,
            valid_fraction
            )
        
    return dataset_dict
    