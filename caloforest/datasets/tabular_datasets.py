# This code is modified from https://github.com/SamsungSAILMontreal/ForestDiffusion
# associated with the paper "Generating and Imputing Tabular Data via Diffusion and Flow-based XGBoost Models"
# by Alexia Jolicoeur-Martineau, Kilian Fatras, and Tal Kachman

import zipfile
import os
import wget
import gzip
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split


def get_tabular_datasets(cfg, dataset, data_root, valid_fraction):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes 
    datasets are located in 'data_root'. If the called for dataset is not in 
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.
        
    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """

    if not os.path.isdir(data_root):
        os.mkdir(data_root)

    bin_y = False # binary outcome
    cat_y = False # categorical w/ >=2 outcome
    int_y = False # integer outcome

    bin_x = None # binary
    cat_x = None # categorical w/ >=2 classes
    int_x = None # integers

    if dataset == 'iris':
        my_data = load_iris()
        cat_y = True
    elif dataset == 'wine':
        my_data = load_wine()
        cat_y = True
    elif dataset == 'california':
        my_data = fetch_california_housing()
        int_x = [1, 4]
    elif dataset == 'parkinsons':
        my_data = fetch_parkinsons(data_root)
        bin_y = True
    elif dataset == 'climate_model_crashes':
        my_data = fetch_climate_model_crashes(data_root)
        bin_y = True
    elif dataset == 'concrete_compression':
        my_data = fetch_concrete_compression(data_root)
        int_x = [7]
    elif dataset == 'yacht_hydrodynamics':
        my_data = fetch_yacht_hydrodynamics(data_root)
    elif dataset == 'airfoil_self_noise':
        my_data = fetch_airfoil_self_noise(data_root)
    elif dataset == 'connectionist_bench_sonar':
        my_data = fetch_connectionist_bench_sonar(data_root)
        bin_y = True
    elif dataset == 'ionosphere':
        my_data = fetch_ionosphere(data_root)
        bin_x = [0]
        bin_y = True
    elif dataset == 'qsar_biodegradation':
        my_data = fetch_qsar_biodegradation(data_root)
        int_x = [2,3,4,5,6,8,9,10,15,18,19,20,22,25,31,32,33,34,37,39,40]
        bin_x = [23,24,28]
        bin_y = True
    elif dataset == 'seeds':
        my_data = fetch_seeds(data_root)
        cat_y = True
    elif dataset == 'glass':
        my_data = fetch_glass(data_root)
        cat_y = True
    elif dataset == 'ecoli':
        my_data = fetch_ecoli(data_root)
        cat_y = True
    elif dataset == 'yeast':
        my_data = fetch_yeast(data_root)
        cat_y = True
    elif dataset == 'libras':
        my_data = fetch_libras(data_root)
        cat_y = True
    elif dataset == 'planning_relax':
        my_data = fetch_planning_relax(data_root)
        bin_y = True
    elif dataset == 'blood_transfusion':
        my_data = fetch_blood_transfusion(data_root)
        int_x = [0,1,3]
        bin_y = True
    elif dataset == 'breast_cancer_diagnostic':
        my_data = fetch_breast_cancer_diagnostic(data_root)
        bin_y = True
    elif dataset == 'connectionist_bench_vowel':
        my_data = fetch_connectionist_bench_vowel(data_root)
        bin_y = True
    elif dataset == 'concrete_slump':
        my_data = fetch_concrete_slump(data_root)
    elif dataset == 'wine_quality_red':
        int_y = True
        my_data = fetch_wine_quality_red(data_root)
    elif dataset == 'wine_quality_white':
        int_y = True
        my_data = fetch_wine_quality_white(data_root)
    elif dataset == 'bean':
        my_data = fetch_bean(data_root)
        int_x = [0,6]
        cat_y = True
    elif dataset == 'tictactoe': # all categorical
        my_data = fetch_tictactoe(data_root)
        cat_x = [0,1,2,3,4,5,6,7,8]
        bin_y = True
    elif dataset == 'congress': # all categorical
        my_data = fetch_congress(data_root)
        cat_x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        cat_y = True
    elif dataset == 'car': # all categorical
        my_data = fetch_car(data_root)
        cat_x = [0,1,2,3,4,5]
        cat_y = True
    elif dataset == 'higgs':
        my_data = fetch_higgs(data_root)
        bin_y = True
    elif dataset == 'random':
        my_data = fetch_random(cfg['num_random_datapoints'], cfg['num_random_features'], cfg['num_random_classes'], cfg['seed'])
        cat_y = True
    else:
        raise Exception('dataset does not exists')
    X, y = my_data['data'], my_data['target']

    # Split data
    seed = cfg["seed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y if bin_y or cat_y else None)
    valid_size = int(valid_fraction * X_train.shape[0])
    if valid_fraction >= 1.0:
        raise ValueError(f"valid_fraction cannot be greater than 1. Was {valid_fraction}")
    elif valid_fraction < 0.0:
        valid_fraction = 0.0
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(X_train.shape[0])
    shuffled_X = X_train[perm]
    shuffled_y = y_train[perm]
    X_valid = shuffled_X[:valid_size]
    y_valid = shuffled_y[:valid_size]
    X_train = shuffled_X[valid_size:]
    y_train = shuffled_y[valid_size:]

    dataset_dict = {}
    dataset_dict["train"] = [X_train, y_train]
    dataset_dict["valid"] = [X_valid, y_valid]
    dataset_dict["test"] = [X_test, y_test]
    dataset_dict["cat_indexes"] = [cat_x, cat_y]
    dataset_dict["bin_indexes"] = [bin_x, bin_y]
    dataset_dict["int_indexes"] = [int_x, int_y]

    return dataset_dict


def fetch_parkinsons(data_root):
    if not os.path.isdir(f'{data_root}parkinsons'):
        os.mkdir('data/parkinsons')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        wget.download(url, out='data/parkinsons/')

    with open('data/parkinsons/parkinsons.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = 0)
        Xy = {}
        Xy['data'] = np.concatenate((df.values[:, 1:17].astype('float'), df.values[:, 18:].astype('float')), axis=1)
        Xy['target'] =  pd.factorize(df.values[:, 17])[0] # str to numeric

    return Xy


def fetch_climate_model_crashes(data_root):
    if not os.path.isdir('data/climate_model_crashes'):
        os.mkdir('data/climate_model_crashes')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat'
        wget.download(url, out='data/climate_model_crashes/')

    with open('data/climate_model_crashes/pop_failures.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, 2:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_concrete_compression(data_root):
    if not os.path.isdir(f'{data_root}concrete_compression'):
        os.mkdir(f'{data_root}concrete_compression')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        wget.download(url, out=f'{data_root}concrete_compression/')

    with open(f'{data_root}concrete_compression/Concrete_Data.xls', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy


def fetch_yacht_hydrodynamics(data_root):
    if not os.path.isdir(f'{data_root}yacht_hydrodynamics'):
        os.mkdir(f'{data_root}yacht_hydrodynamics')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
        wget.download(url, out=f'{data_root}yacht_hydrodynamics/')

    with open(f'{data_root}yacht_hydrodynamics/yacht_hydrodynamics.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy

def fetch_airfoil_self_noise(data_root):
    if not os.path.isdir(f'{data_root}airfoil_self_noise'):
        os.mkdir(f'{data_root}airfoil_self_noise')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        wget.download(url, out=f'{data_root}airfoil_self_noise/')

    with open(f'{data_root}airfoil_self_noise/airfoil_self_noise.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy


def fetch_connectionist_bench_sonar(data_root):
    if not os.path.isdir(f'{data_root}connectionist_bench_sonar'):
        os.mkdir(f'{data_root}connectionist_bench_sonar')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        wget.download(url, out=f'{data_root}connectionist_bench_sonar/')

    with open(f'{data_root}connectionist_bench_sonar/sonar.all-data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_ionosphere(data_root):
    if not os.path.isdir(f'{data_root}ionosphere'):
        os.mkdir(f'{data_root}ionosphere')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out=f'{data_root}ionosphere/')

    with open(f'{data_root}ionosphere/ionosphere.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = np.concatenate((df.values[:, 0:1].astype('float'), df.values[:, 2:-1].astype('float')), axis=1) # removing the secon variable which is always 0
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_qsar_biodegradation(data_root):
    if not os.path.isdir(f'{data_root}qsar_biodegradation'):
        os.mkdir(f'{data_root}qsar_biodegradation')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out=f'{data_root}qsar_biodegradation/')

    with open(f'{data_root}qsar_biodegradation/biodeg.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_seeds(data_root):
    if not os.path.isdir(f'{data_root}seeds'):
        os.mkdir(f'{data_root}seeds')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
        wget.download(url, out=f'{data_root}seeds/')

    with open(f'{data_root}seeds/seeds_dataset.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1] - 1 # make 0, 1, 2 instead of 1, 2, 3

    return Xy


def fetch_glass(data_root):
    if not os.path.isdir(f'{data_root}glass'):
        os.mkdir(f'{data_root}glass')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        wget.download(url, out=f'{data_root}glass/')

    with open(f'{data_root}glass/glass.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  (df.values[:, -1] - 1).astype('int') # make 0, 1, 2 instead of 1, 2, 3
        Xy['target'][Xy['target'] >= 4] = Xy['target'][Xy['target'] >= 4] - 1 # 0, 1, 2, 4, 5, 6 -> 0, 1, 2, 3, 4, 5

    return Xy


def fetch_ecoli(data_root):
    if not os.path.isdir(f'{data_root}ecoli'):
        os.mkdir(f'{data_root}ecoli')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
        wget.download(url, out=f'{data_root}ecoli/')

    with open(f'{data_root}ecoli/ecoli.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy

def fetch_yeast(data_root):
    if not os.path.isdir(f'{data_root}yeast'):
        os.mkdir(f'{data_root}yeast')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        wget.download(url, out=f'{data_root}yeast/')

    with open(f'{data_root}yeast/yeast.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_libras(data_root):
    if not os.path.isdir(f'{data_root}libras'):
        os.mkdir(f'{data_root}libras')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
        wget.download(url, out=f'{data_root}libras/')

    with open(f'{data_root}libras/movement_libras.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1] - 1  # make 0, 1, 2 instead of 1, 2, 3

    return Xy

def fetch_planning_relax(data_root):
    if not os.path.isdir(f'{data_root}planning_relax'):
        os.mkdir(f'{data_root}planning_relax')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt'
        wget.download(url, out=f'{data_root}planning_relax/')

    with open(f'{data_root}planning_relax/plrx.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1] - 1

    return Xy


def fetch_blood_transfusion(data_root):
    if not os.path.isdir(f'{data_root}blood_transfusion'):
        os.mkdir(f'{data_root}blood_transfusion')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        wget.download(url, out=f'{data_root}blood_transfusion/')

    with open(f'{data_root}blood_transfusion/transfusion.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_breast_cancer_diagnostic(data_root):
    if not os.path.isdir(f'{data_root}breast_cancer_diagnostic'):
        os.mkdir(f'{data_root}breast_cancer_diagnostic')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out=f'{data_root}breast_cancer_diagnostic/')

    with open(f'{data_root}breast_cancer_diagnostic/wdbc.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] = pd.factorize(df.values[:, 1])[0] # str to numeric

    return Xy


def fetch_connectionist_bench_vowel(data_root):
    if not os.path.isdir(f'{data_root}connectionist_bench_vowel'):
        os.mkdir(f'{data_root}connectionist_bench_vowel')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        wget.download(url, out=f'{data_root}connectionist_bench_vowel/')

    with open(f'{data_root}connectionist_bench_vowel/vowel-context.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 3:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_concrete_slump(data_root):
    if not os.path.isdir(f'{data_root}concrete_slump'):
        os.mkdir(f'{data_root}concrete_slump')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
        wget.download(url, out=f'{data_root}concrete_slump/')

    with open(f'{data_root}concrete_slump/slump_test.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, 1:-3].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float') # the 3 last variables are actually outcomes, but we choose 1, because we can't have 3!

    return Xy


def fetch_wine_quality_red(data_root):
    if not os.path.isdir(f'{data_root}wine_quality_red'):
        os.mkdir(f'{data_root}wine_quality_red')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        wget.download(url, out=f'{data_root}wine_quality_red/')

    with open(f'{data_root}wine_quality_red/winequality-red.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy


def fetch_wine_quality_white(data_root):
    if not os.path.isdir(f'{data_root}wine_quality_white'):
        os.mkdir(f'{data_root}wine_quality_white')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        wget.download(url, out=f'{data_root}wine_quality_white/')

    with open(f'{data_root}wine_quality_white/winequality-white.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy


def fetch_bean(data_root):
    if not os.path.isdir(f'{data_root}DryBeanDataset'):
        os.mkdir(f'{data_root}DryBeanDataset')
        url = 'https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip'
        wget.download(url, out=f'{data_root}DryBeanDataset/')

    with zipfile.ZipFile(f'{data_root}DryBeanDataset/dry+bean+dataset.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{data_root}')

    with open(f'{data_root}DryBeanDataset/Dry_Bean_Dataset.xlsx', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_tictactoe(data_root):
    if not os.path.isdir(f'{data_root}tictactoe'):
        os.mkdir(f'{data_root}tictactoe')
        url = 'https://archive.ics.uci.edu/static/public/101/tic+tac+toe+endgame.zip'
        wget.download(url, out=f'{data_root}tictactoe/')

    with zipfile.ZipFile(f'{data_root}tictactoe/tic+tac+toe+endgame.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{data_root}tictactoe')

    with open(f'{data_root}tictactoe/tic-tac-toe.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] =  np.zeros(df.values[:, :-1].shape)
        for i in range(Xy['data'].shape[1]):
            Xy['data'][:, i] = pd.factorize(df.values[:, i])[0]
        Xy['target'] =  pd.factorize(df.values[:, -1])[0]

    return Xy


def fetch_congress(data_root):
    if not os.path.isdir(f'{data_root}congress'):
        os.mkdir(f'{data_root}congress')
        url = 'https://archive.ics.uci.edu/static/public/105/congressional+voting+records.zip'
        wget.download(url, out=f'{data_root}congress/')

    with zipfile.ZipFile(f'{data_root}congress/congressional+voting+records.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{data_root}congress')

    with open(f'{data_root}congress/house-votes-84.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] =  np.zeros(df.values[:, 1:].shape)
        for i in range(Xy['data'].shape[1]):
            Xy['data'][:, i] = pd.factorize(df.values[:, i+1])[0]
        Xy['target'] =  pd.factorize(df.values[:, 0])[0]

    return Xy


def fetch_car(data_root):
    if not os.path.isdir(f'{data_root}car'):
        os.mkdir(f'{data_root}car')
        url = 'https://archive.ics.uci.edu/static/public/19/car+evaluation.zip'
        wget.download(url, out=f'{data_root}car/')

    with zipfile.ZipFile(f'{data_root}car/car+evaluation.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{data_root}car')

    with open(f'{data_root}car/car.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] =  np.zeros(df.values[:, :-1].shape)
        for i in range(Xy['data'].shape[1]):
            Xy['data'][:, i] = pd.factorize(df.values[:, i])[0]
        Xy['target'] =  pd.factorize(df.values[:, -1])[0]

    return Xy


def fetch_higgs(data_root):
    if not os.path.isdir(f'{data_root}higgs'):
        os.mkdir(f'{data_root}higgs')
        url = 'https://archive.ics.uci.edu/static/public/280/higgs.zip'
        wget.download(url, out=f'{data_root}higgs/')

        with zipfile.ZipFile(f'{data_root}higgs/higgs.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{data_root}higgs')

    with gzip.open(f'{data_root}higgs/HIGGS.csv.gz', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, 1:].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, 0])[0] # str to numeric

    return Xy


def fetch_random(n, p, n_y, seed):
    rng = np.random.default_rng(seed)
    n = int(n / 0.8) # training dataset is only 80% of all data.
    arr_X = rng.random(size=(n, p), dtype=np.float32)
    arr_y = rng.integers(low=0, high=n_y, size=n, dtype=np.int32)
    Xy = {}
    Xy['data'] = arr_X
    Xy['target'] = arr_y

    return Xy
