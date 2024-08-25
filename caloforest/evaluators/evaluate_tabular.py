import numpy as np
import pandas as pd
import xgboost as xgb
import ot as pot
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, r2_score, pairwise_distances
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

from .utils import dummify, minmax_scale_dummy


OTLIM = 5000 # Max optimal transport steps


def test_on_multiple_models(X_train, y_train, X_test, y_test, cat_indexes=[], classifier=False, nexp=5):
    simplefilter("ignore", category=ConvergenceWarning)
    f1_score_lin = 0.0
    f1_score_linboost = 0.0
    f1_score_tree = 0.0
    f1_score_treeboost = 0.0
    R2_score_lin = 0.0
    R2_score_linboost = 0.0
    R2_score_tree = 0.0
    R2_score_treeboost = 0.0

    # Dummify categorical variables (>=3 categories)
    n = X_train.shape[0]
    if len(cat_indexes) > 0:
        X_train_test, _, _ = dummify(np.concatenate((X_train, X_test), axis=0), cat_indexes)
        X_train_ = X_train_test[0:n,:]
        X_test_ = X_train_test[n:,:]
    else:
        X_train_ = X_train
        X_test_ = X_test

    for i in range(nexp):
        if classifier:
            if not np.isin(np.unique(y_test), np.unique(y_train)).all(): # not enough classes were generated, score=0
                f1_score_lin += 0
                f1_score_linboost += 0
                f1_score_tree += 0
                f1_score_treeboost += 0
            else:
                # Ensure labels go from 0 to num_classes, otherwise it wont work
                # Assumes uniques(y_train) >= uniques(y_test)
                le = LabelEncoder()
                y_train_ = le.fit_transform(y_train)

                y_pred = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500, random_state=i).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
                f1_score_lin += f1_score(y_test, y_pred, average='macro') / nexp

                y_pred = AdaBoostClassifier(random_state=i).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
                f1_score_linboost += f1_score(y_test, y_pred, average='macro') / nexp

                y_pred = RandomForestClassifier(max_depth=28, random_state=i).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
                f1_score_tree += f1_score(y_test, y_pred, average='macro') / nexp

                y_pred = xgb.XGBClassifier(reg_lambda=0.0, random_state=i).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
                f1_score_treeboost += f1_score(y_test, y_pred, average='macro') / nexp
        else:
            y_pred = LinearRegression().fit(X_train_, y_train).predict(X_test_)
            R2_score_lin += r2_score(y_test, y_pred) / nexp

            y_pred = AdaBoostRegressor(random_state=i).fit(X_train_, y_train).predict(X_test_)
            R2_score_linboost += r2_score(y_test, y_pred) / nexp

            y_pred = RandomForestRegressor(max_depth=28, random_state=i).fit(X_train_, y_train).predict(X_test_)
            R2_score_tree += r2_score(y_test, y_pred) / nexp

            y_pred = xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0.0, random_state=i).fit(X_train_, y_train).predict(X_test_)
            R2_score_treeboost += r2_score(y_test, y_pred) / nexp

    f1_score_mean = (f1_score_lin + f1_score_linboost + f1_score_tree + f1_score_treeboost) / 4
    R2_score_mean = (R2_score_lin + R2_score_linboost + R2_score_tree + R2_score_treeboost) / 4

    return {'mean': f1_score_mean, 'lin': f1_score_lin, 'linboost': f1_score_linboost, 'tree': f1_score_tree, 'treeboost': f1_score_treeboost}, {'mean': R2_score_mean, 'lin': R2_score_lin, 'linboost': R2_score_linboost, 'tree': R2_score_tree, 'treeboost': R2_score_treeboost}


# Metrics from https://stefvanbuuren.name/fimd/sec-evaluation.html
# Raw bias and coverage rate (truth is considered to be the regression result from training on the oracle non-missing data)
def test_imputation_regression(X_oracle, y_oracle, X_fake, y_fake, cat_indexes=[], type_model='regression'):
    n = X_oracle.shape[0]

    # Dummy coding (Not one-hot) of the categorical variables (>=3 categories)
    if cat_indexes and len(cat_indexes) > 0:
        X_oracle_, _, _ = dummify(X_oracle, cat_indexes, drop_first=True)
        X_both, _, _ = dummify(np.concatenate((X_oracle, X_fake), axis=0), cat_indexes, drop_first=True)
        X_fake_ = X_both[n:,:]
    else:
        X_oracle_ = X_oracle
        X_fake_ = X_fake

    p = X_oracle_.shape[1]

    # Oracle results
    X_oracle_pd = pd.DataFrame(sm.tools.tools.add_constant(X_oracle_), columns = [str(i) for i in range(p+1)])
    y_oracle_pd = pd.DataFrame(y_oracle, columns = ['y'])
    if type_model == 'multiclass':
        fit_oracle = sm.MNLogit(y_oracle_pd, X_oracle_pd).fit()
    elif type_model == 'class':
        fit_oracle = sm.Logit(y_oracle_pd, X_oracle_pd).fit()
    else: # regression
        fit_oracle = sm.OLS(y_oracle_pd, X_oracle_pd).fit()
    params_oracle = fit_oracle.params.to_numpy()

    percent_bias = np.zeros(params_oracle.shape) # |(E(Q)-Q)/Q|
    coverage_rate = np.zeros(params_oracle.shape) # Percent of times the 95% confidence interval contains the true parameters
    AW = np.zeros(params_oracle.shape) # Average width of the confidence interval
    average_params = np.zeros(params_oracle.shape)

	# Imputation results
    if sm.tools.tools.add_constant(X_fake_).shape[1] == X_fake_.shape[1]:
        print('Warning: Found constant generated variable')
    else:
        X_fake_pd = pd.DataFrame(sm.tools.tools.add_constant(X_fake_), columns = [str(i) for i in range(p+1)])
        y_fake_pd = pd.DataFrame(y_fake, columns = ['y'])
        if type_model == 'multiclass':
            fit_fake = sm.MNLogit(y_fake_pd, X_fake_pd).fit()
        elif type_model == 'class':
            fit_fake = sm.Logit(y_fake_pd, X_fake_pd).fit()
        else: # regression
            fit_fake = sm.OLS(y_fake_pd, X_fake_pd).fit()
        average_params = fit_fake.params.to_numpy()
        coverage_rate = ((params_oracle > fit_fake.conf_int(0.05).to_numpy()[:,0].reshape(params_oracle.shape)) & (params_oracle < fit_fake.conf_int(0.05).to_numpy()[:,1].reshape(params_oracle.shape)))
        AW += (fit_fake.conf_int(0.05).to_numpy()[:,1].reshape(params_oracle.shape) -  fit_fake.conf_int(0.05).to_numpy()[:,0].reshape(params_oracle.shape))

    percent_bias = 100*np.abs((average_params - params_oracle) / params_oracle)

    return percent_bias.mean(), coverage_rate.mean(), AW.mean()

## Below is coverage from https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
def compute_pairwise_distance(data_x, data_y=None): # Changed to L1 instead of L2 to better handle mixed data
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = pairwise_distances(
        data_x, data_y, metric='cityblock', n_jobs=-1)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_coverage(real_features, fake_features, nearest_k=None):
    """
    Computes coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        coverage: float.
    """

    if nearest_k is None: # we choose k to be the smallest such that we have 95% coverage with real data
        coverage_ = 0
        nearest_k = 1
        while coverage_ < 0.95:
            coverage_ = compute_coverage(real_features, real_features, nearest_k)
            nearest_k += 1

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return coverage


def evaluate_tabular_samples(X_gen, y_gen, dataset_dict, cfg):
    X_train, y_train = dataset_dict["train"]
    X_test, y_test = dataset_dict["test"]
    cat_x, cat_y = dataset_dict["cat_indexes"]
    int_x, int_y = dataset_dict["int_indexes"]
    bin_x, bin_y = dataset_dict["bin_indexes"]
    # Create joint versions of datasets
    Xy_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1).astype(X_train.dtype, copy=False)), axis=1)
    Xy_test = np.concatenate((X_test, np.expand_dims(y_test, axis=1).astype(X_test.dtype, copy=False)), axis=1)
    Xy_gen = np.concatenate((X_gen, np.expand_dims(y_gen, axis=1).astype(X_gen.dtype, copy=False)), axis=1)

    cat_indexes = []
    if cat_x is not None:
        cat_indexes = cat_indexes + cat_x
    if cat_y:
        cat_indexes.append(X_train.shape[1])

    # Create scaled versions of datasets
    Xy_train_scaled, Xy_gen_scaled, _, _, _ = minmax_scale_dummy(Xy_train, Xy_gen, cat_indexes, divide_by=2)
    _, Xy_test_scaled, _, _, _ = minmax_scale_dummy(Xy_train, Xy_test, cat_indexes, divide_by=2)

    metrics_dict = {}
    # Wasserstein-1 based on L1 cost (after scaling)
    if Xy_train.shape[0] < OTLIM:
        metrics_dict["score_W1_train"] = pot.emd2(pot.unif(Xy_train_scaled.shape[0]),
                                                  pot.unif(Xy_gen_scaled.shape[0]),
                                                  M = pot.dist(Xy_train_scaled,
                                                  Xy_gen_scaled,
                                                  metric='cityblock'))
        metrics_dict["score_W1_test"] = pot.emd2(pot.unif(Xy_test_scaled.shape[0]),
                                                 pot.unif(Xy_gen_scaled.shape[0]),
                                                 M = pot.dist(Xy_test_scaled,
                                                 Xy_gen_scaled,
                                                 metric='cityblock'))
    f1 = {}
    R2 = {}
    # Trained on train data
    f1["train"], R2["train"] = test_on_multiple_models(X_train, y_train, X_test, y_test, classifier=cat_y or bin_y)
    # Trained on gen data
    f1["gen"], R2["gen"] = test_on_multiple_models(X_gen, y_gen, X_test, y_test, classifier=cat_y or bin_y)
    # Trained on train and gen data
    X_both = np.concatenate((X_train, X_gen), axis=0)
    y_both = np.concatenate((y_train, y_gen))
    f1["both"], R2["both"] = test_on_multiple_models(X_both, y_both, X_test, y_test, classifier=cat_y or bin_y)

    for key in ['mean', 'lin', 'linboost', 'tree', 'treeboost']:
        for data in ["train", "gen", "both"]:
            metrics_dict[f"f1_{data}_{key}"] = f1[data][key]
            metrics_dict[f"R2_{data}_{key}"] = R2[data][key]

    # coverage based on L1 cost (after scaling)    
    metrics_dict["coverage"] = compute_coverage(Xy_train_scaled, Xy_gen_scaled, None)
    metrics_dict["coverage_test"] = compute_coverage(Xy_test_scaled, Xy_gen_scaled, None)
    # # Statistical measures
    # Too unstable with classification due toquasi-seperation with logistic regression
    # dataset=ecoli with missing data is removed because it has near-constant variables, and the non-constant parts can be lost when adding missing data making it perfectly multicorrelated which will give regression errors
    if not cat_y and not bin_y and not (cfg["dataset"] == 'ecoli'):
        percent_bias_, coverage_rate_, AW_ = test_imputation_regression(X_train, y_train, X_gen, y_gen, 
            cat_indexes=cat_x, type_model='regression')
    else: 
        percent_bias_, coverage_rate_, AW_ = 0.0, 0.0, 0.0

    metrics_dict["percent_bias"] = percent_bias_
    metrics_dict["coverage_rate"] = coverage_rate_
    metrics_dict["AW"] = AW_
    
    return metrics_dict
