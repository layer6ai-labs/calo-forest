import os

DATASETS = ['iris', 'wine', 'california', 'parkinsons', \
            'climate_model_crashes', 'concrete_compression', \
            'yacht_hydrodynamics', 'airfoil_self_noise', \
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
            'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', \
            'blood_transfusion', 'breast_cancer_diagnostic', \
            'connectionist_bench_vowel', 'concrete_slump', \
            'wine_quality_red', 'wine_quality_white', \
            'bean', 'tictactoe','congress','car']

# Writing to file
# UPDATE THIS
home_dir = os.environ['HOME']
script_file_path = home_dir + "/research/calo-forest/scripts/"
results_file_path = "saved_runs/"

# Number of CPU available
# UPDATE THIS
n_cpu = 40

# seed
seed = 0

# Long running datasets should log usage less frequently
long_runs = ['bean', 'connectionist_bench_sonar', 'libras', 'qsar_biodegradation']
log_delay = 1.0

# Baseline - match hyperparameters of ForestDiffusion.
# This matches model performance of Original, but uses our implementation for speed and memory benefits.
with open(script_file_path + "run_original.sh", "w") as file1:
    path = results_file_path + "original"
    for seed_i in range(0, 3):
        seed_i = seed_i + seed
        for dataset in DATASETS:
            log_delay = 10.0 if dataset in long_runs else 1.0
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=flow --config scaler=single_min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0 --config multinomial=True --config logdir_root={path}\n")
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=vp --config scaler=single_min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0.001 --config multinomial=True --config logdir_root={path}\n")
            file1.write("\n")
        
# Use single-output trees with class-conditional scalers.
with open(script_file_path + "run_SO.sh", "w") as file1:
    path = results_file_path + "SO"
    for seed_i in range(0, 3):
        seed_i = seed_i + seed
        for dataset in DATASETS:
            log_delay = 10.0 if dataset in long_runs else 1.0
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=flow --config scaler=min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0 --config logdir_root={path}\n")
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=vp --config scaler=min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0.001 --config logdir_root={path}\n")
            file1.write("\n")

# Use multi-output trees.
with open(script_file_path + "run_MO.sh", "w") as file1:
    path = results_file_path + "MO"
    for seed_i in range(0, 3):
        seed_i = seed_i + seed
        for dataset in DATASETS:
            log_delay = 10.0 if dataset in long_runs else 1.0
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=flow --config multi_strategy=multi_output_tree --config scaler=min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0 --config logdir_root={path}\n")
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=vp  --config multi_strategy=multi_output_tree --config scaler=min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0.001 --config logdir_root={path}\n")
            file1.write("\n")

# Use early stopping and scaled up hyperparameters.
with open(script_file_path + "run_SO_Scaled.sh", "w") as file1:
    path = results_file_path + "SO_Scaled"
    for seed_i in range(0, 3):
        seed_i = seed_i + seed
        for dataset in DATASETS:
            log_delay = 10.0 if dataset in long_runs else 1.0
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=flow --config early_stopping_rounds=20 --config duplicate_K=1000 --config eta=0.3 --config n_estimators=2000 --config scaler=min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0 --config logdir_root={path}\n")
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=vp --config early_stopping_rounds=20 --config duplicate_K=1000 --config eta=0.3 --config n_estimators=2000 --config scaler=min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0.001 --config logdir_root={path}\n")
            file1.write("\n")

# Use multi-output trees, early stopping, and scaled up hyperparameters.
with open(script_file_path + "run_MO_Scaled.sh", "w") as file1:
    path = results_file_path + "MO_Scaled"
    for seed_i in range(0, 3):
        seed_i = seed_i + seed
        for dataset in DATASETS:
            log_delay = 10.0 if dataset in long_runs else 1.0
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=flow --config multi_strategy=multi_output_tree --config early_stopping_rounds=20 --config duplicate_K=1000 --config eta=0.3 --config n_estimators=2000 --config scaler=min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0 --config logdir_root={path}\n")
            file1.write(f"python main.py --dataset={dataset} --config diffusion_type=vp --config multi_strategy=multi_output_tree --config early_stopping_rounds=20 --config duplicate_K=1000 --config eta=0.3 --config n_estimators=2000 --config scaler=min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed_i} --config log_delay={log_delay} --config eps=0.001 --config logdir_root={path}\n")
            file1.write("\n")

# Resource Scaling Experiments
# Must use the original repository to measure their performace.
# Create random datasets of different sizes.
with open(script_file_path + "run_resource_scaling.sh", "w") as file1:
    path = results_file_path + "resource_scaling"
    multi_strategy = ['one_output_per_tree', 'multi_output_tree']
    num_datapoints = [1000, 3000, 10000, 30000, 100000, 300000]
    base_datapoints_idx = 0
    num_features = [3, 10, 30, 100, 300]
    base_features_idx = 1
    num_classes = [1, 3, 10, 30, 100]
    base_classes_index = 2

    for method in multi_strategy:
        for n in num_datapoints:
            file1.write(f"python main.py --skip-eval --dataset=random --config num_random_datapoints={n} --config num_random_features={num_features[base_features_idx]} --config num_random_classes={num_classes[base_classes_index]} --config multi_strategy={method} --config diffusion_type=flow --config scaler=single_min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed} --config log_delay=1.0 --config eps=0 --config logdir_root={path}/{method}\n")
        file1.write("\n")
        base = num_features.pop(base_features_idx) # this would be a duplicate run
        for n in num_features:
            file1.write(f"python main.py --skip-eval --dataset=random --config num_random_datapoints={num_datapoints[base_datapoints_idx]} --config num_random_features={n} --config num_random_classes={num_classes[base_classes_index]} --config multi_strategy={method} --config diffusion_type=flow --config scaler=single_min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed} --config log_delay=1.0 --config eps=0 --config logdir_root={path}/{method}\n")
        file1.write("\n")
        num_features.insert(base_features_idx, base)
        base = num_classes.pop(base_classes_index) # this would be a duplicate run
        for n in num_classes:
            file1.write(f"python main.py --skip-eval --dataset=random --config num_random_datapoints={num_datapoints[base_datapoints_idx]} --config num_random_features={num_features[base_features_idx]} --config num_random_classes={n} --config multi_strategy={method} --config diffusion_type=flow --config scaler=single_min_max --config n_jobs={n_cpu} --config xgb_n_jobs=1 --config seed={seed} --config log_delay=1.0 --config eps=0 --config logdir_root={path}/{method}\n")
        file1.write("\n")
        num_classes.insert(base_classes_index, base)