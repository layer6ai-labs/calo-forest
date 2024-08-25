# This code and package is heavily modified from https://github.com/SamsungSAILMontreal/ForestDiffusion
# associated with the paper "Generating and Imputing Tabular Data via Diffusion and Flow-based XGBoost Models"
# by Alexia Jolicoeur-Martineau, Kilian Fatras, and Tal Kachman

import os
import tempfile
import gc
import shutil
import copy
import numpy as np
import xgboost as xgb
import pandas as pd

from functools import partial
from sklearn.preprocessing import MinMaxScaler
from joblib import delayed, Parallel, load, dump, cpu_count
from tqdm import tqdm

from caloforest.forest_diffusion.utils import build_data_single_xt, output_to_predict, get_solver_fn, build_data_iterator, VPSDE, get_pc_sampler


## Class for the flow-matching or diffusion model
# Categorical features should be numerical (rather than strings), make sure to use x = pd.factorize(x)[0] to make them as such
# Make sure to specify which features are categorical and which are integers
# Note: Binary features can be considered integers since they will be rounded to the nearest integer and then clipped
class ForestModel():

    def __init__(self,
                 n_t=50, # number of noise levels
                 diffusion_type='vp', # vp or flow
                 xgb_hypers={},
                 duplicate_K=100, # number of different noise samples per real data sample
                 cat_indexes=[], # vector which indicates which column is categorical (>=3 categories)
                 bin_indexes=[], # vector which indicates which column is binary
                 int_indexes=[], # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
                 true_min_max_values=None, # Vector of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
                 eps=0.001, # timestep to stop generation at, often used with diffusion models which can explode as t->0.
                 beta_min=0.1, # vp only
                 beta_max=8, # vp only
                 solver="euler", # euler, heun, or rk4
                 scaler="min_max", # min_max creates one scaler per class. single_min_max creates one scaler overall.
                 n_jobs=-1, # number of parallel jobs to create. xgb_hypers contains xgb_n_jobs, which is the number of cpus per job.
                 backend="loky", # joblib Parallel backend. Can be "loky", "multiprocessing", or "threading". We recommend not changing this.
                 n_batch=-1, # If >0, use data iterator with the specified number of batches when constructing QuantileDMatrix
                 seed=0,
                 logdir=".",
                 ):

        self.cat_indexes = cat_indexes if cat_indexes else []
        bin_indexes = bin_indexes if bin_indexes else []
        int_indexes = int_indexes if int_indexes else []
        self.int_indexes = int_indexes + bin_indexes # since we round those, we do not need to one-hot encode the binary variables
        self.true_min_max_values = true_min_max_values
        self.n_t = n_t
        self.duplicate_K = duplicate_K
        self.xgb_hypers = xgb_hypers
        self.seed = seed
        self.n_jobs = n_jobs
        self.backend = backend
        self.set_logdir(logdir)
        assert diffusion_type == 'vp' or diffusion_type == 'flow'
        self.diffusion_type = diffusion_type
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sde = None
        if diffusion_type == 'vp':
            self.sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t, seed=self.seed)
        self.set_solver_fn(solver)
        self.scaler_type = scaler
        self.n_batch = n_batch
        self.use_xgb_core_api = (n_batch > 0)

    def set_logdir(self, logdir):
        self.logdir = logdir
        self.train_dir = os.path.join(logdir, "train")
        self.checkpoint_path = os.path.join(logdir, "TRAIN_CHECKPOINT.txt")

    def preprocess(self,
                   X,
                   label_y=None # must be a categorical/binary variable; if provided will learn multiple models for each label y
                   ):
        np.random.seed(self.seed)
        # Remove observations with only missing data
        obs_to_remove = np.isnan(X).all(axis=1)
        X = X[~obs_to_remove]
        self.trained_with_y = label_y is not None
        if self.trained_with_y:
            label_y = label_y[~obs_to_remove]
        if self.true_min_max_values is not None:
            self.X_min = self.true_min_max_values[0]
            self.X_max = self.true_min_max_values[1]
        else:
            self.X_min = np.nanmin(X, axis=0, keepdims=1)
            self.X_max = np.nanmax(X, axis=0, keepdims=1)

        # One-hot encoding for categorical variables
        if len(self.cat_indexes) > 0:
            X, self.X_names_before, self.X_names_after = self.dummify(X)
        self.n, self.p = X.shape

        # Switch to XGBoost's native dtype
        X0 = X.astype(np.float32, copy=True)
        # Create masks for each class y
        if self.trained_with_y:
            # Cannot have missing values in the label (just make a special categorical for nan if you need)
            assert np.sum(np.isnan(label_y)) == 0 
            # Sort X0 according to label_y
            y_arg_sort = np.argsort(label_y)
            label_y = label_y[y_arg_sort]
            X0 = X0[y_arg_sort]
            # Create slices to mask sorted data by class
            self.y_uniques, y_counts = np.unique(label_y, return_counts=True)
            self.y_probs = y_counts / np.sum(y_counts)
            self.mask_y = {} # mask for which observations has a specific value of y
            csum = 0
            for y, count in zip(self.y_uniques, y_counts):
                self.mask_y[y] = slice(csum, csum + count)
                csum += count
        else: # assuming a single unique label 0
            self.y_probs = np.array([1.0])
            self.y_uniques = np.array([0])
            self.mask_y = {0: slice(0, self.n)} # single mask covering all data

        # Data normalization to ensure that data is in the range [-1, 1]
        # which is on the same scale as noise added from a standard Normal.
        if self.scaler_type in ["single_min_max", "min_max"]:
            scaler_kwargs = {'feature_range': (-1, 1)}
            if self.scaler_type == "single_min_max":
                print("Scaling all data with MinMaxScaler")
                self.scaler = MinMaxScaler(**scaler_kwargs)
                X0 = self.scaler.fit_transform(X0)
            else:
                print("Scaling each class with MinMaxScaler")
                self.scalers = {}
                for y in self.y_uniques:
                    self.scalers[y] = MinMaxScaler(**scaler_kwargs)
                    X0[self.mask_y[y], :] = self.scalers[y].fit_transform(X0[self.mask_y[y], :])
        else:
            print("Not scaling the data")
        
        return X0

    def train(self, X0):
        # directory path to save models
        os.makedirs(self.train_dir, exist_ok=True)

        if self.n_batch <= 0:
            # Duplicate the data K times, each will be associated with different noise samples
            X0 = np.repeat(X0, self.duplicate_K, axis=0) # Data
            rng = np.random.default_rng(self.seed)
            X1 = rng.standard_normal(size=X0.shape, dtype=X0.dtype) # Noise
            Z = output_to_predict(X0, X1, diffusion_type=self.diffusion_type) # For VP and Flow, regression target is constant.
            if self.xgb_hypers["early_stopping_rounds"] is not None:
                X1_valid = rng.standard_normal(size=X0.shape, dtype=X0.dtype) # Noise data for validation
                Z_valid = output_to_predict(X0, X1_valid, diffusion_type=self.diffusion_type)

        ts = np.linspace(self.eps, 1, num=self.n_t) # for diffusion we may stop generation before t=0

        # Improve cpu utilization for the last njobs; schedule dynamically inside train_xgb_parallel()
        njobs = self.n_jobs if self.n_jobs > 0 else cpu_count()
        xgb_n_jobs = self.xgb_hypers['n_jobs'] if self.xgb_hypers['n_jobs'] > 0 else cpu_count()
        total_cpu_use = min(njobs * xgb_n_jobs, cpu_count())
        ntasks = self.n_t * len(self.mask_y)

        def train_xgb_parallel(X0_mmap, X1_mmap, X1_valid_mmap, Z_mmap, Z_valid_mmap,
                               sl, t, i, j, build_data, params, model_dir, n_batch):
            path = os.path.join(model_dir, f"model_{i}_{j}.ubj")
            if (os.path.exists(path)): # the model is already trained
                return

            # Improve cpu utilization for the last njobs
            if params['n_jobs'] > 0 and params['device'] == 'cpu':
                n_incomplete_tasks = ntasks - len(os.listdir(model_dir))
                if n_incomplete_tasks < njobs:
                    params['n_jobs'] = int(total_cpu_use / n_incomplete_tasks + 0.9) # almost ceiling

            if n_batch <= 0:
                X_train = build_data(X0_mmap[sl], X1_mmap[sl], t)
                Z_train = Z_mmap[sl]
                if params['early_stopping_rounds'] is not None:
                    X_valid = build_data(X0_mmap[sl], X1_valid_mmap[sl], t)
                    Z_valid = Z_valid_mmap[sl]
                    eval_set = [(X_valid, Z_valid)]
                else:
                    eval_set = None
                out = xgb.XGBRegressor(**params)
                out.fit(X_train, Z_train, eval_set=eval_set, verbose=False)
                out.save_model(path)

            else: # use data iterator to construct QuantileDMatrix
                it = build_data(X0_mmap[sl], t, params['seed'])
                train_dmat = xgb.QuantileDMatrix(it)
                params_copy = params.copy() # shallow copy is fine
                n_boost_round = params_copy.pop('n_estimators')
                es_rounds = params_copy.pop('early_stopping_rounds')
                if es_rounds is not None:
                    it_valid = build_data(X0_mmap[sl], t, params['seed'] + 666)
                    evals = [(xgb.QuantileDMatrix(it_valid, ref=train_dmat), "valid")]
                else:
                    evals = None
                out = xgb.train(params_copy, train_dmat, num_boost_round=n_boost_round,
                                early_stopping_rounds=es_rounds, evals=evals, verbose_eval=False)
                out.save_model(path)
            return

        xgb_params = {
            'objective': 'reg:squarederror',
            'subsample': 1.0,
            'seed': self.seed,
            'tree_method': 'hist',
            **self.xgb_hypers,
        }

        if self.n_batch <= 0:
            build_data_f = partial(build_data_single_xt, diffusion_type=self.diffusion_type, sde=self.sde)
        else:
            build_data_f = partial(
                build_data_iterator,
                n_batch=self.n_batch,
                duplicate_K=self.duplicate_K,
                diffusion_type=self.diffusion_type,
                sde=self.sde
            )
        
        train_parallel = partial(
            train_xgb_parallel,
            build_data=build_data_f,
            params=xgb_params,
            model_dir=self.train_dir,
            n_batch=self.n_batch
        )

        def create_memmap(array, file_path):
            dump(array, file_path)
            mmap = load(file_path, mmap_mode='r')
            return mmap

        # create memory-mapped files and free memory for X0, X1, Z
        # memory-mapped files are cached in shared memory and can be freed upon memory pressure
        temp_folder = tempfile.mkdtemp()
        X0_mmap = create_memmap(X0, os.path.join(temp_folder, 'X0.mmap'))
        del(X0)
        X1_mmap = None
        Z_mmap = None
        X1_valid_mmap = None
        Z_valid_mmap = None
        if self.n_batch <= 0:
            X1_mmap = create_memmap(X1, os.path.join(temp_folder, 'X1.mmap'))
            Z_mmap = create_memmap(Z, os.path.join(temp_folder, 'Z.mmap'))
            del(X1, Z)
            if xgb_params["early_stopping_rounds"] is not None:
                X1_valid_mmap = create_memmap(X1_valid, os.path.join(temp_folder, 'X1_valid.mmap'))
                Z_valid_mmap = create_memmap(Z_valid, os.path.join(temp_folder, 'Z_valid.mmap'))
                del(X1_valid, Z_valid)
        gc.collect()

        y_slice = {}
        if self.n_batch <= 0:
            for y, sl in self.mask_y.items():
                y_slice[y] = slice(sl.start*self.duplicate_K, sl.stop*self.duplicate_K)
        else:
            for y, sl in self.mask_y.items():
                y_slice[y] = slice(sl.start, sl.stop)  

        # Fit model(s)
        try:
            with Parallel(n_jobs=self.n_jobs, backend=self.backend, max_nbytes=None, return_as="generator_unordered") as parallel:
                with tqdm(total=self.n_t * len(y_slice)) as pbar:
                    res = parallel(
                        delayed(train_parallel)(
                            X0_mmap, X1_mmap, X1_valid_mmap, Z_mmap, Z_valid_mmap,
                            y_slice[j], ts[i], i, j,
                        ) for i in range(self.n_t) for j in range(len(y_slice))
                    )
                    n_complete_tasks = 0
                    for _ in res:
                        n_complete_tasks += 1
                        with open(self.checkpoint_path, 'w') as file:
                            file.write(f"Complete training {n_complete_tasks} ensembles out of {ntasks}\n")
                        pbar.update(1)
        finally:
            shutil.rmtree(temp_folder, ignore_errors=False)

        with open(self.checkpoint_path, 'w') as file:
            file.write(f"Done\n")

    def load_models(self):
        # check if the models are already loaded
        if hasattr(self, "regr") and self.regr is not None:
            return

        # for backward compatibility
        if not hasattr(self, "use_xgb_core_api"):
            self.use_xgb_core_api = False

        # check training is finished
        train_not_done = True
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as file:
                checkpoint = file.readline()
                if "Done" in checkpoint:
                    train_not_done = False
        if train_not_done:
            raise RuntimeError("Try to load models but training is not completed.")

        self.regr = [[None for _ in range(self.n_t)] for _ in self.y_uniques]

        def load_single_model(i, j):
            booster = xgb.Booster() if self.use_xgb_core_api else xgb.XGBRegressor()
            booster.load_model(os.path.join(self.train_dir, f"model_{i}_{j}.ubj"))
            return booster

        # parallel loading with 4 jobs because sequential model loading only achieves
        # 25% average cpu utilization. This could be different on different machines
        regr_ = Parallel(n_jobs=4, backend="threading")(
            delayed(load_single_model)(
                i, j
            ) for i in range(self.n_t) for j in self.y_uniques
        )
        current_i = 0
        for i in range(self.n_t):
            for j in range(len(self.y_uniques)):
                self.regr[j][i] = regr_[current_i]
                current_i += 1

    def dummify(self, X):
        df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
        df_names_before = df.columns
        for i in self.cat_indexes:
            df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=True)
        df_names_after = df.columns
        df = df.to_numpy()
        return df, df_names_before, df_names_after

    def unscale(self, X, mask_y):
        if self.scaler_type == "single_min_max":
            X = self.scaler.inverse_transform(X)
        elif self.scaler_type == "min_max":
            for k, mask in mask_y.items():
                X[mask, :] = self.scalers[k].inverse_transform(X[mask, :])
        return X

    # Rounding for the categorical variables which are dummy-coded and then remove dummy-coding
    def clean_onehot_data(self, X):
        if len(self.cat_indexes) > 0: # ex: [5, 3] and X_names_after [gender_a gender_b cartype_a cartype_b cartype_c]
            X_names_after = copy.deepcopy(self.X_names_after.to_numpy())
            prefixes = [x.split('_')[0] for x in self.X_names_after if '_' in x] # for all categorical variables, we have prefix ex: ['gender', 'gender']
            unique_prefixes = np.unique(prefixes) # uniques prefixes
            for i in range(len(unique_prefixes)):
                cat_vars_indexes = [unique_prefixes[i] + '_' in my_name for my_name in self.X_names_after]
                cat_vars_indexes = np.where(cat_vars_indexes)[0] # actual indexes
                cat_vars = X[:, cat_vars_indexes] # [n, p_cat]
                # dummy variable, so third category is true if all dummies are 0
                cat_vars = np.concatenate((np.ones((cat_vars.shape[0], 1))*0.5,cat_vars), axis=1)
                # argmax of -1, -1, 0 is 0; so as long as they are below 0 we choose the implicit-final class
                max_index = np.argmax(cat_vars, axis=1) # argmax across all the one-hot features (most likely category)
                X[:, cat_vars_indexes[0]] = max_index
                X_names_after[cat_vars_indexes[0]] = unique_prefixes[i] # gender_a -> gender
            df = pd.DataFrame(X, columns = X_names_after) # to Pandas
            df = df[self.X_names_before] # remove all gender_b, gender_c and put everything in the right order
            X = df.to_numpy()
        return X

    # Unscale and clip to prevent going beyond min-max and also round off the integers
    def clip_extremes(self, X):
        X_min = X.dtype.type(self.X_min)
        X_max = X.dtype.type(self.X_max)
        if self.int_indexes is not None:
            for i in self.int_indexes:
                X[:,i] = np.round(X[:,i], decimals=0)
        small = (X < X_min).astype(X.dtype)
        X = small*X_min + (1-small)*X
        big = (X > X_max).astype(X.dtype)
        X = big*X_max + (1-big)*X
        return X

    # Return the score-fn or ode-flow output
    def my_model_j(self, t, x, j, xgb_njobs=-1):
        i = int(round((t-self.eps) / (1-self.eps) * (self.n_t-1)))
        booster = self.regr[j][i]
        if self.use_xgb_core_api:
            booster.set_param('n_jobs', xgb_njobs)
            # When a model is trained with early stopping, there is an inconsistent behavior
            # between native Python interface and sklearn interfaces. By default on sklearn
            # interfaces, the best_iteration is automatically used so prediction comes from
            # the best model. But with the native Python interface xgboost.Booster.predict()
            # and xgboost.Booster.inplace_predict() uses the full model.
            # https://xgboost.readthedocs.io/en/stable/prediction.html#early-stopping
            if hasattr(booster, "best_iteration"):
                out = booster.inplace_predict(x, iteration_range=(0, booster.best_iteration))
            else:
                out = booster.inplace_predict(x)
        else:
            booster.get_booster().set_param('n_jobs', xgb_njobs)
            out = booster.predict(x)
        if self.diffusion_type == 'vp':
            alpha_, sigma_ = self.sde.marginal_prob_coef(x, t)
            out = - out / sigma_
        return out

    # Generate new data by solving the reverse ODE/SDE
    def generate(self, batch_size=None, n_t=None, label_y=None, seed=None):
        self.load_models()

        # Generate prior noise
        if seed is None:
            seed = self.seed + 1 # different seeds should be used for training and generation
        elif seed == self.seed:
                raise ValueError("""You may be using the same noise samples for generation as used in training.\n
              This can lead to non-representative results, or higher likelihood to generate training data.\n
              Consider changing the generation seed.""")
        rng = np.random.default_rng(seed)
        X1 = rng.standard_normal(size=(self.n if batch_size is None else batch_size, self.p), dtype=np.float32)

        # If labels not provided for conditional generation,
        # sample labels from a multinomial with probabilities taken from training data
        if label_y is None:
            label_y_used = self.y_uniques[np.argmax(rng.multinomial(1, self.y_probs, size=X1.shape[0]), axis=1)]
        else:
            label_y_used = label_y
        y_sort_idx = np.argsort(label_y_used)
        label_y_used = label_y_used[y_sort_idx]
        y_uniques, y_counts = np.unique(label_y_used, return_counts=True)
        mask_y = {} # mask for which observations has a specific value of y
        csum = 0
        for y, count in zip(y_uniques, y_counts):
            mask_y[y] = slice(csum, csum + count)
            csum += count

        # allocate more cpu to bigger samples
        ncpu = cpu_count()
        y_ratio = np.empty(y_counts.shape, dtype=np.float32)
        for i in range(0, y_counts.size, ncpu):
            y_ratio[i:i+ncpu] = y_counts[i:i+ncpu] * ncpu / np.sum(y_counts[i:i+ncpu])
        y_ratio = np.round(y_ratio)
        y_ratio = y_ratio.astype(np.int32)
        y_ratio[y_ratio == 0] = 1
        xgb_njobs = {}
        for y, r in zip(y_uniques, y_ratio):
            xgb_njobs[y] = r

        if self.diffusion_type == 'vp':
            def run_sampler(x1, my_model, sde, eps):
                return get_pc_sampler(my_model, sde=sde, denoise=True, eps=eps)(x1)
            
            sde_solved = Parallel(n_jobs=-1, backend="threading", batch_size=1)(
                delayed(run_sampler)(
                    X1[mask_y[j]],
                    partial(self.my_model_j, j=j, xgb_njobs=xgb_njobs[j]),
                    VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.n_t if n_t is None else n_t, seed=seed),
                    self.eps,
                ) for j in y_uniques
            )
            solution = np.concatenate(sde_solved, axis=0)
        elif self.diffusion_type == 'flow':
            n_t = self.n_t if n_t is None else n_t
            ode_solved = Parallel(n_jobs=-1, backend="threading", batch_size=1)(
                delayed(self.solver_fn)(
                    X1[mask_y[j]],
                    partial(self.my_model_j, j=j, xgb_njobs=xgb_njobs[j]),
                    n_t,
                    self.eps,
                ) for j in y_uniques
            )
            solution = np.concatenate(ode_solved, axis=0)
        else:
            raise ValueError(f"diffusion_type {self.diffusion_type} not implemented.")

        solution = self.unscale(solution, mask_y)
        solution = self.clean_onehot_data(solution)
        solution = self.clip_extremes(solution)
        
        unsort_idx = np.argsort(y_sort_idx)
        solution = solution[unsort_idx]

        # Concatenate y label if model was trained with X and y separate
        if self.trained_with_y:
            label_y_used = label_y_used[unsort_idx]
            solution = np.concatenate((solution, np.expand_dims(label_y_used, axis=1).astype(solution.dtype, copy=False)), axis=1)

        return solution

    def set_solver_fn(self, solver):
        self.solver_fn = get_solver_fn(solver)
