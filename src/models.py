# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty 
import dataclasses
from typing import Optional, Union

import jax.numpy as jnp 
import jax.random as jnpr 
import numpy as np 

from io_utils import human_bytes_str
from solvers import least_squares

pytree: type = Union[dict, jnp.ndarray, list]

def last_value(x: jnp.ndarray) -> jnp.ndarray: 
    """Simple last-value predictor"""
    return x[-1]

def running_average(params: dict, x: jnp.ndarray) -> jnp.ndarray: 
    params["sum"] += x.sum()
    params["count"] += x.size
    return params, params["sum"] / params["count"]

class RunningAverage: 
    def __init__(self): 
        self.params = {"sum": 0.0, "count": 0.0}

    def fit(self, inputs: np.ndarray, targets: np.ndarray): 
        pass

    def evaluate(self, inputs: np.ndarray) -> np.ndarray: 
        predictions = [] 

        for x in inputs: 
            self.params, prediction = running_average(self.params, x)
            predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions

def linear_ar_model(params: pytree, x: jnp.ndarray) -> jnp.ndarray: 
    return params["weights"] @ x + params["bias"]

class ForecastingModel(ABC): 
    @property 
    @abstractmethod 
    def parameters(self) -> Optional[np.ndarray]: 
        raise NotImplementedError

    @abstractmethod 
    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None: 
        raise NotImplementedError

    @abstractmethod 
    def __call__(self, inputs: np.ndarray) -> np.ndarray: 
        raise NotImplementedError

@dataclasses.dataclass 
class LinearAutoregressiveModelConfig: 
    memory: int 
    rollout: int 
    norm_penalty: Optional[float]=0.0

class LinearAutoregressiveModel(ForecastingModel): 
    def __init__(self, config: LinearAutoregressiveModelConfig): 
        self.config = config 
        self._parameters: np.ndarray = None

    def __repr__(self) -> str: 
        return f"{self.__class__.__name__}(config={self.config}, initialized={self.parameters is not None}, size={human_bytes_str(self.num_bytes)})"

    @property 
    def parameters(self) -> Optional[np.ndarray]: 
        return self._parameters

    @property 
    def num_bytes(self) -> int: 
        return self.parameters.nbytes if self.parameters is not None else 0

    def fit(self, inputs: np.ndarray, targets: np.ndarray): 
        num_samples: int = inputs.shape[0]
        num_variables: int = inputs.shape[1]
        model_memory: int = inputs.shape[2]

        A: np.ndarray = inputs.reshape(num_samples, num_variables * model_memory)
        b: np.ndarray = targets.reshape(num_samples, num_variables)

        if self.config.norm_penalty > 0.0:
            penalty: np.ndarray = self.config.norm_penalty * np.eye(num_variables * model_memory)
            A: np.ndarray = np.concatenate((A, penalty))
            b: np.ndarray = np.concatenate((b, np.zeros((num_variables * model_memory, num_variables))))

        self._parameters: np.ndarray = least_squares(A, b)

    def generate(self, prefix: np.ndarray, num: int) -> np.ndarray: 
        raise NotImplementedError

    def __call__(self, inputs: np.ndarray) -> np.ndarray: 
        def _forward(inputs: np.ndarray) -> np.ndarray: 
            if self.parameters is None: 
                raise ValueError("Parameters not initialized... please call `fit` with problem data.")

            num_variables, model_memory = inputs.shape[-2:]
            inputs: np.ndarray = inputs.reshape(-1, num_variables * model_memory)
            return (inputs @ self.parameters).reshape(inputs.shape[:-2] + (num_variables,))
        return np.vectorize(_forward, signature='(m,n)->(m)')(inputs)
