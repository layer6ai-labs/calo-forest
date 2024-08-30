<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Diffusion and Flow-based XGBoost Models

This is the codebase accompanying the work ["Scaling Up Diffusion and Flow-based XGBoost Models"](https://arxiv.org/abs/2408.16046). It builds on [ForestDiffusion](https://github.com/SamsungSAILMontreal/ForestDiffusion) associated with the paper ["Generating and Imputing Tabular Data via Diffusion and Flow-based XGBoost Models"](https://arxiv.org/abs/2309.09968). Here we discuss how to run the experiments in the paper and give a general overview of the codebase.

## Python Environment Setup

The main prerequisite is to set up the python environment.
The command

    conda env create -f environment.yml

will create a `conda` environment called `caloforest`.
Launch this environment using the command

    conda activate caloforest

before running any experiments.

For experiments involving multi-output trees, versions of XGBoost prior to v2.1.0 have an incorrect gain computation which [we identified](https://github.com/dmlc/xgboost/issues/9960). Hence, this codebase should be run with XGB v2.1.0 or greater.

## Datasets 

This codebase is designed for experimentation on the datasets from the ["Fast Calorimeter Simulation Challenge 2022"](https://calochallenge.github.io/homepage/), which contain simulated calorimeter showers produced by incident photons, pions. These datasets can be downloaded at the following links:
[Photons](https://zenodo.org/records/8099322)
[Pions](https://zenodo.org/records/6366271)

27 other datasets are implemented and are listed in `get_config.py`. Additionally, we implemented synthetic random datasets of controllable size under the name `random`.

## Usage - `main.py`

The main script for running experiments is unsurprisingly `main.py`. The basic usage is as follows:

    python main.py --dataset <dataset>

where `<dataset>` is the dataset, one of "photons1", "pions1", "random", or one of the 27 other options mentioned above.

### Dynamic Updating of Config Values

Model and training hyperparameters are loaded from the config files in the directory `config` at runtime. For each hyperparameter `<key>` that one wants to set to a new value `<value>`, add the following to the command line:

    --config <key>=<value>

We can do this multiple times for the config. An example is 

    python main.py --dataset iris --config duplicate_K=1000 --config n_estimators=2000

A full list of config values is visible in the `base_config.py` file in the `config` directory. Notably, multi-output trees are run with

    --config multi_strategy=multi_output_tree

Resource usage logging is activated with

    --config log_delay=<time in seconds>

Models are trained on CPU by default. GPU training is implemented (`--config device=gpu`), but not optimized.

### Run Directories

By default, the `main` command above will create a directory of the form `runs/<date>_<hh>-<mm>-<ss>`, e.g. `Apr26_09-38-37`, to store information about the run, including:

- Model checkpoints
- Experiment metrics / results as `json` or `csv`
- Config files as `json`
- `stderr` / `stdout` logs
- Plots of resource usage if the `log_delay` option is enabled.

We provide the ability to reload a saved run with run directory `<dir>` via the command:

    ./main.py --load-dir <dir>

which will restart the training of the model (if not completed) and then perform testing.

## Usage - `main-sample-tab.py` and `main-sample-calo.py`

If one has a trained model, use one of these files to load it and generate new samples. the `tab` file is for the tabular datasets, while the `calo` file is for Photons and Pions. The `--evaluate` flag will also compute metrics on the generated samples.

    ./main-sample-tab.py --load-dir <dir> --evaluate

## Scripts to run experiments

You can easily generate scripts that will reproduce our experiments by first running

    python script_generator.py

You may need to first update the directory information in that file.

## Create Singularity Image

The caloforest.def file can be used to build a runnable singularity image by calling

    sudo singularity build <desired_image_name>.sif caloforest.def

You will first need to install `go` (tested with version `go1.20.7 linux/amd64`) and `singularity` (tested with version `3.11.4`). Once the image is built, it can be run with

    singularity run <desired_image_name>.sif

## Citing

If you use any part of this repository in your research, please cite the associated paper with the following bibtex entry:

```
@article{cresswell2024scaling,
  title={Scaling Up Diffusion and Flow-based XGBoost Models},
  author={Cresswell, Jesse C and Kim, Taewoo},
  journal={arXiv:2408.16046},
  year={2024}
}
```

## License

This data and code is licensed under the MIT License, copyright by Layer 6 AI.
