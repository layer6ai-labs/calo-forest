# This code and package is heavily modified from https://github.com/SamsungSAILMontreal/ForestDiffusion
# associated with the paper "Generating and Imputing Tabular Data via Diffusion and Flow-based XGBoost Models"
# by Alexia Jolicoeur-Martineau, Kilian Fatras, and Tal Kachman
import numpy as np
import xgboost as xgb

def output_to_predict(x0, x1, diffusion_type):
    if diffusion_type == 'vp':
        z = x1
    elif diffusion_type == 'flow':
        z = x1 - x0 # [n, p]
    else:
        raise ValueError(f"diffusion_type {diffusion_type} not implemented.")
    return z

def build_data_single_xt(x0, x1, t, diffusion_type, sde=None):
    if diffusion_type == 'vp': # Forward diffusion from x0 to x1
        mean, std = sde.marginal_prob(x0, t)
        x_t = mean + std*x1
    elif diffusion_type == 'flow': # Interpolation between x0 and x1
        x_t = t * x1 + (1 - t) * x0 # [n, p]
    else:
        raise ValueError(f"diffusion_type {diffusion_type} not implemented.")
    return x_t

# Get X[t], y where t is a scalar
def get_xt(x0, t, rng, diffusion_type='flow', sde=None):
    x1 = rng.standard_normal(size=x0.shape, dtype=x0.dtype) # Noise
    x_t = build_data_single_xt(x0, x1, t, diffusion_type, sde)
    z = output_to_predict(x0, x1, diffusion_type)
    return x_t, z

# Seperate dataset into multiple batches for memory-efficient QuantileDMatrix construction
class IterForDMatrix(xgb.core.DataIter):
    """A data iterator for XGBoost DMatrix.

    `reset` and `next` are required for any data iterator, other functions here
    are utilites for demonstration's purpose.

    """

    def __init__(self, data, t, seed, n_batch=1, n_epochs=100, diffusion_type='flow', sde=None):
        self._data = data
        self.seed = seed
        self.n_batch = n_batch
        n = self._data.shape[0]
        rows_per_batch = n // self.n_batch
        self.batch_slices = [slice(i * rows_per_batch, (i+1) * rows_per_batch) for i in range(self.n_batch)]
        if self.n_batch * rows_per_batch != n:
            self.batch_slices.append(slice(self.n_batch * rows_per_batch, n))
            self.n_batch += 1
        self.n_epochs = n_epochs
        self.t = t
        self.diffusion_type = diffusion_type
        self.sde = sde
        self.it = 0  # set iterator to 0
        self.seed = seed
        super().__init__()
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self):
        """Reset the iterator"""
        self.it = 0
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

    def next(self, input_data):
        """Yield next batch of data."""
        if self.it == self.n_batch*self.n_epochs: # stops after k epochs
            return 0
        x_t, z = get_xt(x0=self._data[self.batch_slices[self.it % self.n_batch]], t=self.t, rng=self.rng, diffusion_type=self.diffusion_type, sde=self.sde)
        input_data(data=x_t, label=z)
        self.it += 1
        return 1

def build_data_iterator(x0, t, seed, n_batch, duplicate_K, diffusion_type, sde):
    return IterForDMatrix(x0, t, seed, n_batch=n_batch, n_epochs=duplicate_K, diffusion_type=diffusion_type, sde=sde)

#### Below is for Flow-Matching Sampling ####

# Euler solver
def euler_solve(x1, my_model, N=101, eps=0.0):
    h = (1-eps) / (N-1)
    x = x1
    t = 1
    # from t=1 to t=eps
    for i in range(N-1):
        x = x - h*my_model(t=t, x=x)
        t = t - h
    return x

def heun_solve(x1, my_model, N=101, eps=0.0):
    h = (1-eps) / (N-1)
    x = x1
    t = 1
    # from t=1 to t=eps
    for i in range(N-1):
        slope_t = my_model(t=t, x=x)
        x_tilde = x - h*slope_t
        t = t - h
        x = x - h*(slope_t + my_model(t=t, x=x_tilde))/2
    return x

def rk4_solve(x1, my_model, N=101, eps=0.0):
    N_half = (N-1)//2
    h = (1-eps) / (N-1)
    x = x1
    t = 1
    # from t=1 to t=eps
    for i in range(N_half):
        slope_1 = my_model(t=t, x=x)
        slope_2 = my_model(t=t-h, x=x-h*slope_1)
        slope_3 = my_model(t=t-h, x=x-h*slope_2)
        slope_4 = my_model(t=t-2*h, x=x-2*h*slope_3)
        x = x - h*(slope_1 + 2*slope_2 + 2*slope_3 + slope_4)/3
        t = t - 2*h
    return x

def get_solver_fn(solver):
    if solver == "euler":
        return euler_solve
    elif solver == "heun":
        return heun_solve
    elif solver == "rk4":
        return rk4_solve
    else:
        raise ValueError(f"Solver {solver} not implemented")
