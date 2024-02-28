# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import sys 

sys.path.insert(0, "/forecaster/src")
sys.path.insert(0, "/pyslds")
sys.path.insert(0, "/pybasicbayes")

import dataclasses
from pathlib import Path 
from typing import Optional

from matplotlib.cm import get_cmap 
from matplotlib.gridspec import GridSpec 
import matplotlib.pyplot as plt 
import numpy as np 
import numpy.random as npr 
import jax.numpy as jnp 
from jax import jit, grad, vmap
import jax.random as jnpr 
from statsmodels.tsa.stattools import acf 
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import Lasso, Ridge

from dataset import ForecastingConfig, ForecastingDataset, SegmentationConfig, SegmentationDataset
from metrics import rms, percentage_error
from models import last_value, RunningAverage

from pybasicbayes.distributions import DiagonalRegression, Gaussian, Regression
from pyslds.models import DefaultSLDS, WeakLimitHDPHMMSLDS
from pybasicbayes.util.text import progprint_xrange
from pylds.util import random_rotation

npr.seed(18)

@dataclasses.dataclass 
class SLDSConfig: 
    num_states: Optional[int]=5
    observation_dimension: Optional[int]=1
    latent_dimension: Optional[int]=3
    input_dimension: Optional[int]=0

config = SLDSConfig(
    num_states=2
)

As = [random_rotation(config.latent_dimension) for _ in range(config.num_states)]

model = DefaultSLDS(
    config.num_states, 
    config.observation_dimension, 
    config.latent_dimension, 
    D_input=config.input_dimension, 
    As=As, 
    alpha=1., 
    )

data_config = SegmentationConfig(
    data_source=Path("/forecaster/data/aggregated.pkl"), 
)
segmentation_dataset = SegmentationDataset(data_config) 

data = segmentation_dataset.data[0]
smooth: callable = lambda x, order=5: np.convolve(x, np.ones(order) / order, mode="same")[order:-order]

data = smooth(data[0, :], order=3)

num_timesteps: int = data.shape[-1]
fit_proportion: float = 0.75 
num_train: int = int(num_timesteps * fit_proportion)
x: np.ndarray = data[:num_train]
x_train: np.ndarray = x.copy()
x_test: np.ndarray = data[num_train:]

model.add_data(x[:, None])

num_gibbs_samples: int = 300
def initialize(model): 
    model.resample_model()
    return model.log_likelihood()

gibs_lls: list = [initialize(model) for _ in progprint_xrange(num_gibbs_samples)]

def extract_contiguous_regions(x: np.ndarray) -> list: 
    regions = {}

    unique_values = np.unique(x)

    for unique in unique_values: 
        regions[str(unique)] = [] 

        region = [] 
        previous_end: int = 0 
        for i in range(previous_end, x.shape[-1]): 
            if x[i] == unique: 
                region.append(i)
            else: 
                if region: 
                    regions[str(unique)].append(np.array(region))
                    region = [] 
                previous_end = i 

    return regions


# extract mode datasets
state_sequence = model.states_list[0].stateseq[None, :][0]
regions = extract_contiguous_regions(state_sequence)

from copy import deepcopy

def data_from_regions(data, regions): 
    master_config = ForecastingConfig(
        data_source="", 
        predictors=None, 
        aggregate_benchmarks=True
    )

    mode_identifiers = list(regions.keys())
    mode_datasets = dict()

    for mode in mode_identifiers: 
        _datasets = [] 
        for contiguous_region in regions[mode]: 
            config = deepcopy(master_config)
            if contiguous_region.size < (3 * config.model_memory): 
                pass
            else: 
                _x = data[contiguous_region].copy()
                _t = _x.copy()
                ds = ForecastingDataset(config, lazy=True) 
                ds.from_ndarray(_x[None, :], _t, train_only=True)
                _datasets.append(ds)

        mode_datasets[mode] = _datasets

    # aggregate each mode into one dataset 
    aggregated_datasets = {} 

    for mode in mode_identifiers: 
        if mode_datasets[mode]: 
            aggregated = mode_datasets[mode][0]

            for i in range(1, len(mode_datasets[mode])): 
                aggregated = aggregated.add(mode_datasets[mode][i])

            aggregated_datasets[mode] = aggregated

    return aggregated_datasets

# generate mode specific predictors TODO generalize 
aggregated_datasets = data_from_regions(x_train, regions)
mode_identifiers = list(aggregated_datasets.keys())
mode_predictors = {} 

for mode in mode_identifiers: 
    _data = aggregated_datasets[mode].data
    A: np.ndarray = _data["train"][0][0]
    A = A.reshape(A.shape[0], -1)
    b: np.ndarray = _data["train"][0][1]
    x: np.ndarray = np.linalg.solve(A.T @ A, A.T @ b)
    mode_predictors[mode] = x 

model.temperature = 0.0

test_z = model.heldout_viterbi(x_test[:, None])
test_regions = extract_contiguous_regions(test_z)
test_data = data_from_regions(x_test, test_regions)

mode_identifiers = list(test_data.keys())

for mode in mode_identifiers: 
    _data = test_data[mode].data
    A: np.ndarray = _data["train"][0][0]
    A = A.reshape(A.shape[0], -1)
    b: np.ndarray = _data["train"][0][1]
    # use the mode predictor built from the in-sample data
    print(f"Mode: {mode}\tPercentage Error: {percentage_error(A @ mode_predictors[mode], b)}%")

dummy: int = 5




