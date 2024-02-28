# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import dataclasses 
import enum 
import glob 
import pathlib
from pathlib import Path 
from typing import Dict, List, Optional, Tuple

import numpy as np 

from io_utils import DATA_DIRECTORY
from io_utils import load_object as deserialize
from parser import PMUEvents

@dataclasses.dataclass 
class PreprocessingConfig: 
    minimum_density: Optional[float]=0.001
    minimum_variability: Optional[float]=0.001
    clip_from_start: Optional[float]=0.10
    clip_from_end: Optional[float]=0.10

class Preprocessor: 
    def __init__(self, config: PreprocessingConfig, source_directory: Optional[Path]=DATA_DIRECTORY): 
        self.config = config 
        self.source_directory: Path = source_directory
        self.filter_diagnostics: Dict[str, int] = dict(
            not_variable_enough=0, 
            not_dense_enough=0
        )

    def process(self): 
        stats_files: List[pathlib.Path] = DATA_DIRECTORY.glob("*.pkl")
        stats: List[Dict[str, np.ndarray]] = [deserialize(file) for file in stats_files]

        # filters 
        clip: callable = lambda ndarray: ndarray[int(self.config.clip_from_start * ndarray.size):-int(self.config.clip_from_end * ndarray.size)]
        dense_enough: callable = lambda ndarray: (np.count_nonzero(ndarray) / ndarray.size) > self.config.minimum_density
        rms: callable = lambda ndarray: np.linalg.norm(ndarray) / np.sqrt(ndarray.size)
        variable_enough: callable = lambda ndarray: (rms(np.diff(ndarray)) / rms(ndarray)) > self.config.minimum_variability

        keys: set = set(stats[0].keys()).intersection(*[set(stat.keys()) for stat in stats[1:]])
        retain: bool = True
        retained: set = set()

        # one special case, we must add IPC here and enforce that it's not filtered out 
        keys.add("IPC")
        for workload in stats: 
            workload["IPC"] = workload[PMUEvents.NUM_INSTRUCTIONS_RETIRED.value] / workload[PMUEvents.NUM_CYCLES.value]

        for key in keys: 
            for workload in stats: 
                time_series: np.ndarray = workload[key]
                time_series: np.ndarray = clip(time_series)

                if key == "IPC": 
                    retain = True
                    break 

                if not dense_enough(time_series): 
                    retain = False
                    self.filter_diagnostics["not_dense_enough"] += 1
                    break 

                if not variable_enough(time_series): 
                    retain = False
                    self.filter_diagnostics["not_variable_enough"] += 1
                    break

            if retain: 
                retained.add(key)
            retain = True

        new_stats = [] 

        for workload in stats: 
            new_workload = copy.deepcopy(workload)
            for workload_key in workload.keys(): 
                if workload_key not in retained: 
                    new_workload.pop(workload_key)

            new_stats.append(new_workload)

        return new_stats

def round_to_multiple(x: int, multiple: int) -> int: 
    return x - (x % multiple)

@dataclasses.dataclass
class ForecastingDatasetConfig: 
    """Configuration dataclass for forecasting tasks. 

    Attributes
    ----------
    byte_stream: pathlib.Path 
        path to (serialized) raw data, which should deserialize as a Python dict with string 
        keys encoding feature identifiers, and numpy.ndarray values containing the time series 
        of values associated with that feature. 
    model_memory: int 
        the number of previous time series values used as input to the model to generate a prediction. 
    rollout: int
        the number of time series values (in the future) to be predicted; i.e., the number of autoregressive 
        invocations of the model. 
    visible_inputs: Optional[str] 
        str specifier for which inputs should be used to construct the dataset (i.e., which keys in the simulation 
        result dictionary should be used to populate the inputs). 
    target_identifier: Optional[str]
        str specifier for the key in the simulation result dictionary that should be used as the target signal. 
    winsor_widths: Optional[Tuple[int, int]]
        head and tail widths to remove from the time series. 
    train_proportion: Optional[float]
        proportion of time series to use to fit the model parameters. 
    validation_proportion: Optional[float]  
        proportion of time series to use for model selection. 
    test_proportion: Optional[float] 
        proportion of time series to use for evaluation and reporting. 
    """
    byte_stream: pathlib.Path
    model_memory: int 
    rollout: int 
    visible_inputs: Optional[str] = "all" # TODO enum 
    target_identifier: Optional[str] = "target" # TODO enum 
    winsor_widths: Optional[Tuple[int]] = (0, 0)
    train_proportion: Optional[float] = 0.70
    validation_proportion: Optional[float] = 0.20
    test_proportion: Optional[float] = 0.10 

    def __eq__(self, config) -> bool: 
        return config.model_memory == self.model_memory \
            and config.rollout == self.rollout \
            and config.visible_inputs == self.visible_inputs \
            and config.target_identifier == self.target_identifier

def verify_split_validity(config: ForecastingDatasetConfig): 
    config_sum: float = np.sum([
            config.train_proportion, 
            config.validation_proportion, 
            config.test_proportion
            ])
    config_has_negative: bool = any([x < 0.0 for x in (config.train_proportion, config.validation_proportion, config.test_proportion)])

    if config_has_negative: 
        raise ValueError("Split proportions must be nonnegative.")
    elif abs(config_sum - 1.0) > 1e-8: 
        raise ValueError(f"Sum of split proportions must sum to 1.0, but given propotions summed to {config_sum}")

class ForecastingDataset:
    def __init__(self, config: ForecastingDatasetConfig, lazy_concretization: Optional[bool]=False): 
        self.config = config
        self._evaluation_mode: bool = False

        self._train_inputs: np.ndarray = None
        self._train_targets: np.ndarray = None
        self._validation_inputs: np.ndarray = None
        self._validation_targets: np.ndarray = None
        self._test_inputs: np.ndarray = None
        self._test_targets: np.ndarray = None

        self._initialized: bool = False
        if not lazy_concretization: self.initialize()

    def __repr__(self) -> str: 
        return f"{self.__class__.__name__}(initialized={self._initialized}, size={human_bytes_str(self.num_bytes)}, config={self.config})"

    @property 
    def num_bytes(self) -> int: 
        if not self._initialized: return 0 
        num_bytes_inputs: int = self.train_inputs.nbytes + self.validation_inputs.nbytes + self._test_inputs.nbytes 
        num_bytes_targets: int = self.train_targets.nbytes + self.validation_targets.nbytes + self._test_targets.nbytes
        return num_bytes_inputs + num_bytes_targets

    @property 
    def train_inputs(self) -> np.ndarray: 
        if not self._initialized: self.initialize()
        return self._train_inputs

    @property 
    def validation_inputs(self) -> np.ndarray: 
        if not self._initialized: self.initialize()
        return self._validation_inputs

    @property 
    def test_inputs(self) -> np.ndarray: 
        if not self._initialized: self.initialize()
        assert self._evaluation_mode
        return self._test_inputs

    @property 
    def train_targets(self) -> np.ndarray: 
        if not self._initialized: self.initialize()
        return self._train_targets

    @property 
    def validation_targets(self) -> np.ndarray: 
        if not self._initialized: self.initialize()
        return self._validation_targets

    @property 
    def test_targets(self) -> np.ndarray: 
        if not self._initialized: self.initialize()
        assert self._evaluation_mode
        return self._test_targets

    @property 
    def evaluation_mode(self) -> bool: 
        return self._evaluation_mode

    @evaluation_mode.setter 
    def evaluation_mode(self, enable: bool): 
        self._evaluation_mode = enable 

    def results_to_ndarray(self, simulation_results: Dict[str, np.ndarray]) -> Tuple[np.ndarray]: 
        if self.config.visible_inputs != "all": 
            raise NotImplementedError

        inputs: np.ndarray = np.vstack([x for x in simulation_results.values()])
        targets: np.ndarray = simulation_results[self.config.target_identifier]
        return inputs, targets

    def split(self, inputs: np.ndarray, targets: np.ndarray): 
        verify_split_validity(self.config)

        spacing: int = self.config.model_memory
        spacing_timesteps: int = self.config.model_memory * sum([x != 0. for x in [
            self.config.train_proportion, 
            self.config.validation_proportion
        ]])
        num_timesteps: int = inputs.shape[-1] 
        available_timesteps: int = num_timesteps - self.config.rollout

        num_train: int = round_to_multiple(int(self.config.train_proportion * available_timesteps), spacing)

        # TODO what if no validation proportion but a test propotion? 
        validation_remainder: float = (self.config.validation_proportion / (self.config.validation_proportion + self.config.test_proportion))
        num_validation: int = round_to_multiple(int(validation_remainder * (available_timesteps - (num_train + spacing))), spacing)
        num_test: int = round_to_multiple(int(available_timesteps - (num_train + num_validation + 2 * spacing)), spacing)

        assert (num_train > (self.config.model_memory + self.config.rollout))
        assert (num_validation > (self.config.model_memory + self.config.rollout))
        assert (num_test > (self.config.model_memory + self.config.rollout))

        train_end_index: int = num_train
        validation_start_index: int = num_train + spacing
        validation_end_index: int = validation_start_index + num_validation 
        test_start_index: int = validation_end_index + spacing
        test_end_index: int = test_start_index + num_test
        assert test_end_index <= num_timesteps

        train_target_start_index: int = self.config.model_memory
        train_target_end_index: int = train_end_index + 1
        validation_target_start_index: int = validation_start_index + self.config.model_memory 
        validation_target_end_index: int = validation_end_index + 1
        test_target_start_index: int = test_start_index + self.config.model_memory 
        test_target_end_index: int = test_end_index + 1

        self._train_inputs: np.ndarray = np.stack(np.split(inputs[:, :train_end_index], num_train // self.config.model_memory, axis=-1))
        self._validation_inputs: np.ndarray = np.stack(np.split(inputs[:, validation_start_index:validation_end_index], num_validation // self.config.model_memory, axis=-1))
        self._test_inputs: np.ndarray = np.stack(np.split(inputs[:, test_start_index:test_end_index], num_test // self.config.model_memory, axis=-1))

        # TODO handle rank >= 2 arrays
        _get_subsequences: callable = lambda x, indices: np.take(x, indices[:, np.newaxis] + np.arange(self.config.rollout), axis=1).squeeze().T

        self._train_targets: np.ndarray = _get_subsequences(targets, np.arange(train_target_start_index, train_target_end_index, step=self.config.model_memory))
        self._validation_targets: np.ndarray = _get_subsequences(targets, np.arange(validation_target_start_index, validation_target_end_index, step=self.config.model_memory))
        self._test_targets: np.ndarray = _get_subsequences(targets, np.arange(test_target_start_index, test_target_end_index, step=self.config.model_memory))

    def initialize(self): 
        assert self.config.byte_stream.exists()
        simulation_results: Dict[str, np.ndarray] = deserialize(self.config.byte_stream)
        inputs, targets = self.results_to_ndarray(simulation_results)
        self.split(inputs, inputs)
        self._initialized = True

    def __add__(self, dataset): 
        assert dataset.config == self.config

        sum = ForecastingDataset(self.config, lazy_concretization=True)

        breakpoint()
        sum._train_inputs = np.vstack((self.train_inputs, dataset.train_inputs))
        sum._validation_inputs = np.vstack((self.validation_inputs, dataset.validation_inputs))
        sum._test_inputs = np.vstack((self.test_inputs, dataset.test_inputs))

        sum._train_targets = np.vstack((self.train_targets, dataset.train_targets))
        sum._validation_targets = np.vstack((self.validation_targets, dataset.validation_targets))
        sum._test_targets = np.vstack((self.test_targets, dataset.test_targets))
        sum._initialized = True 
        return sum 

if __name__=="__main__": 
    # preprocessing 
    preprocessing_config: PreprocessingConfig = PreprocessingConfig()
    preprocessor: Preprocessor = Preprocessor(preprocessing_config)
    stats: List[Dict[str, np.ndarray]] = preprocessor.process()

    # configure dataset 
    config = ForecastingDatasetConfig()
    dataset = ForecastingDataset(config)
    dataset.initialize(stats)
