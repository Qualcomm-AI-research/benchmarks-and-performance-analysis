# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses 
import logging 
from pathlib import Path
from typing import List, Dict, Optional, Tuple 

import numpy as np 

from custom_logging import setup_logger 
from io_utils import deserialize, human_bytes_str

@dataclasses.dataclass
class SegmentationConfig: 
    data_source: Path 

    # aggregation 
    aggregate_benchmarks: Optional[bool]=False 

    # feature collection 
    features: Optional[str]="ipc"

    # preprocessing 
    upsample_factor: Optional[int]=1
    winsorize: Optional[bool]=True
    front_clip_proportion: Optional[float]=0.05
    back_clip_proportion: Optional[float]=0.05
    standardize: Optional[bool]=False

@dataclasses.dataclass
class ForecastingConfig: 
    data_source: Path 

    # model 
    model_memory: Optional[int]=3 
    rollout: Optional[int]=1

    # predictor/target collections 
    predictors: Optional[str]="ipc" 
    targets: Optional[str]="ipc"

    # preprocessing 
    upsample_factor: Optional[int]=1
    winsorize: Optional[bool]=True
    front_clip_proportion: Optional[float]=0.05
    back_clip_proportion: Optional[float]=0.05
    standardize: Optional[bool]=False

    # split 
    aggregate_benchmarks: Optional[bool]=False 
    train_proportion: Optional[float]=0.70
    validation_proportion: Optional[float]=0.20
    test_proportion: Optional[float]=0.10

def load_source(source_path: Path) -> List[Dict[str, np.ndarray]]: 
    """Loads serialized data given a path. The data is assumed to deserialize 
    as a list of dictionaries (one entry per benchmark simulated).
    """

    if not source_path.exists(): 
        raise FileNotFoundError(f"Did not find a file at: {source_path.absolute().as_posix()}.")

    source_data: List[Dict[str, np.ndarray]] = deserialize(source_path.absolute().as_posix())
    return source_data

def round_to_multiple(x: int, multiple: int) -> int: 
    return x - (x % multiple)

def extract_feature_collection(data: Dict[str, np.ndarray], collection: Optional[str]="ipc", standardize: Optional[bool]=True, upsample_factor: Optional[int]=1) -> np.ndarray: 
    if collection == "ipc": 
        signal_identifiers = [
            "system.cpu_cluster.cpus.ipc"
        ]
    elif collection == "memory": 
        signal_identifiers = [
            "system.cpu_cluster.cpus.statIssuedInstType_0::MemRead", 
            "system.cpu_cluster.cpus.statIssuedInstType_0::MemWrite", 
            "system.cpu_cluster.cpus.MemDepUnit__0.insertedLoads", 
            "system.cpu_cluster.cpus.MemDepUnit__0.insertedStores", 
            "system.cpu_cluster.cpus.dcache.overallMissRate::total", 
            "system.cpu_cluster.l2.demandMissRate::cpu_cluster.cpus.data", 
            "system.cpu_cluster.cpus.ipc"
        ]
    elif collection == "operations": 
        signal_identifiers = [
            "system.cpu_cluster.cpus.statIssuedInstType_0::IntAlu", 
            "system.cpu_cluster.cpus.intAluAccesses", 
            "system.cpu_cluster.cpus.numSquashedInsts", 
            "system.cpu_cluster.cpus.timesIdled", 
            "system.cpu_cluster.cpus.branchPred.lookups", 
            "system.cpu_cluster.cpus.branchPred.condPredicted", 
            "system.cpu_cluster.cpus.branchPred.condIncorrect", 
            "system.cpu_cluster.cpus.branchPred.BTBHitRatio", 
            "system.cpu_cluster.cpus.commit.branchMispredicts", 
            "system.cpu_cluster.cpus.decode.idleCycles", 
            "system.cpu_cluster.cpus.ipc"
        ]
    elif collection == "all": 
        signal_identifiers = [
            "system.cpu_cluster.cpus.statIssuedInstType_0::MemRead", 
            "system.cpu_cluster.cpus.statIssuedInstType_0::MemWrite", 
            "system.cpu_cluster.cpus.MemDepUnit__0.insertedLoads", 
            "system.cpu_cluster.cpus.MemDepUnit__0.insertedStores", 
            "system.cpu_cluster.cpus.dcache.overallMissRate::total", 
            "system.cpu_cluster.l2.demandMissRate::cpu_cluster.cpus.data", 
            "system.cpu_cluster.cpus.statIssuedInstType_0::IntAlu", 
            "system.cpu_cluster.cpus.intAluAccesses", 
            "system.cpu_cluster.cpus.numSquashedInsts", 
            "system.cpu_cluster.cpus.timesIdled", 
            "system.cpu_cluster.cpus.branchPred.lookups", 
            "system.cpu_cluster.cpus.branchPred.condPredicted", 
            "system.cpu_cluster.cpus.branchPred.condIncorrect", 
            "system.cpu_cluster.cpus.branchPred.BTBHitRatio", 
            "system.cpu_cluster.cpus.commit.branchMispredicts", 
            "system.cpu_cluster.cpus.decode.idleCycles", 
            "system.cpu_cluster.cpus.ipc"
        ]
    else: 
        raise NotImplementedError(f"Collection identifier {collection} not supported.")

    if standardize: 
        preprocess: callable = lambda x: (x - x.mean()) / x.std()
    else: 
        preprocess: callable = lambda x: x 

    if upsample_factor != 1: 
        upsample: callable = lambda x: x[::upsample_factor]
    else: 
        upsample: callable = lambda x: x 
    return np.vstack([upsample(preprocess(data[identifier])) for identifier in signal_identifiers])

def compute_split_lengths(num_timesteps: int, config) -> List[int]: 
    assert abs(sum([config.train_proportion, config.validation_proportion, config.test_proportion]) - 1.0) <= 1e-5, f"Provided split configuration does not have proportions that sum to 1.0" 
    num_train: int = int(num_timesteps * config.train_proportion)
    num_validation: int = int((num_timesteps - num_train) * (config.validation_proportion / (config.validation_proportion + config.test_proportion)))
    num_test: int = num_timesteps - (num_train + num_validation) 
    return [num_train, num_validation, num_test]
    
def split(data: np.ndarray, split_lengths: List[int]) -> Tuple[np.ndarray, ...]: 
    num_train, num_validation, num_test = split_lengths
    num_timesteps: int = data.shape[-1]
    assert sum(split_lengths) <= num_timesteps 

    if data.ndim > 1: 
        train: np.ndarray = data[:, :num_train]
        validation: np.ndarray = data[:, num_train:num_train + num_validation]
        test: np.ndarray = data[:, num_train + num_validation:sum(split_lengths)]
    else: 
        train: np.ndarray = data[:num_train]
        validation: np.ndarray = data[num_train:num_train + num_validation]
        test: np.ndarray = data[num_train + num_validation:sum(split_lengths)]
    return (train, validation, test)

def extract_observations(predictors: np.ndarray, targets: np.ndarray, config) -> Tuple[np.ndarray]: 
    num_timesteps: int = predictors.shape[-1] 
    available_timesteps: int = num_timesteps - config.rollout
    n: int = round_to_multiple(available_timesteps, config.model_memory)

    if targets.ndim == 1: 
        targets = targets[None, :]

    _get_subsequences: callable = lambda x, indices: np.take(x, indices[:, np.newaxis] + np.arange(config.rollout), axis=1).squeeze().T

    try: 
        predictors: np.ndarray = np.stack(np.split(predictors[:, :n], n // config.model_memory, axis=-1))
    except: 
        breakpoint()
        dummy: int = 5 
    targets: np.ndarray = _get_subsequences(targets, np.arange(config.model_memory, num_timesteps, step=config.model_memory))

    if targets.ndim == 1: 
        targets = targets[:, None]

    return predictors, targets

def preprocess(x: np.ndarray, config) -> np.ndarray: 
    if x.ndim == 1: 
        flat: bool = True
        x = x[None, :]
    else:
        flat: bool = False

    num_features: int = x.shape[0]
    num_timesteps: int = x.shape[-1]

    if config.winsorize: 
        x: np.ndarray = x[:, int(config.front_clip_proportion * num_timesteps):-int(config.back_clip_proportion * num_timesteps)]

    if flat: 
        x = x.ravel()

    return x 

class SegmentationDataset: 
    def __init__(self, config: SegmentationConfig, log: Optional[logging.Logger]=None): 
        self.config = config 
        self.log = setup_logger(__name__) if (log is None) else log 
        self.data = None
        self.setup()

    def __repr__(self) -> str: 
        repr_str: str = f"{self.__class__.__name__}(config={self.config}, initialized={'True' if self.data is not None else 'False'}"
        if self.data is not None: 
            repr_str += f", size={human_bytes_str(self.nbytes)})"
        else: 
            repr_str += ")"
        return repr_str 

    def setup(self): 
        source_path: Path = Path(self.config.data_source)
        source_data: List[Dict[str, np.ndarray]] = load_source(source_path)
        self.log.info(f"Deserialized source data from: {source_path.absolute().as_posix()}")

        per_benchmark_features: List[np.ndarray] = [] 

        for source in source_data: 
            features: np.ndarray = extract_feature_collection(source, self.config.features, self.config.standardize, self.config.upsample_factor)
            features = preprocess(features, self.config)
            per_benchmark_features.append(features)

        if self.config.aggregate_benchmarks: 
            raise NotImplementedError
        else:
            self.data: List[np.ndarray] = per_benchmark_features

    @property 
    def nbytes(self) -> int: 
        nbytes: int = 0 
        if self.data is None: 
            pass
        elif type(self.data) == list: 
            for x in self.data: 
                nbytes += x.nbytes 
        elif type(self.data) == np.ndarray: 
            nbytes += self.data.nbytes
        else: 
            raise ValueError

        return nbytes

class ForecastingDataset: 
    def __init__(self, config: ForecastingConfig, log: Optional[logging.Logger]=None, lazy: Optional[bool]=False): 
        self.config = config 
        self.log = setup_logger(__name__) if (log is None) else log 
        
        if not lazy: 
            self.log.info("Starting setup...")
            self.setup()
            self.log.info("Finished setup...")

    def __repr__(self) -> str: 
        repr_str: str = f"{self.__class__.__name__}(config={self.config}, initialized={'True' if self.data is not None else 'False'}"

        if self.data is not None: 
            repr_str += f", size={human_bytes_str(self.nbytes)})"
        else: 
            repr_str += ")"
        return repr_str 

    def add(self, dataset): 
        # assumes we're coming from Dict[str, List[Tuple[ndarray, ...]]] representation 
        train = [] 
        validation = [] 
        test = [] 

        train_predictors = np.vstack((self.data["train"][0][0], dataset.data["train"][0][0]))
        train_targets = np.vstack((self.data["train"][0][1], dataset.data["train"][0][1]))

#        validation_predictors = np.vstack((self.data["validation"][0][0], dataset.data["validation"][0][0]))
#        validation_targets = np.vstack((self.data["validation"][0][1], dataset.data["validation"][0][1]))
#
#        test_predictors = np.vstack((self.data["test"][0][0], dataset.data["test"][0][0]))
#        test_targets = np.vstack((self.data["test"][0][1], dataset.data["test"][0][1]))

        train.append((train_predictors, train_targets))
#        validation.append((validation_predictors, validation_targets))
#        test.append((test_predictors, test_targets))
#        new_data = dict(train=train, validation=validation, test=test)
        new_data = dict(train=train)
        self.old_data = self.data
        self.data = new_data
        return self 

    @property 
    def nbytes(self) -> int: 
        nbytes: int = 0 

        if self.data is None: 
            pass
        elif type(self.data) == dict: 
            for x in self.data.values(): 
                if type(x) == np.ndarray: 
                    nbytes += x.nbytes 
        else: 
            pass

        return nbytes

    def from_ndarray(self, predictors: np.ndarray, targets: np.ndarray, train_only: Optional[bool]=False): 
        # preprocessing
        preprocess(predictors, self.config)
        preprocess(targets, self.config)

        num_timesteps: int = predictors.shape[-1]
        if not train_only: 
            train, validation, test = [], [], []

            # split the time series
            split_lengths: List[int] = compute_split_lengths(num_timesteps, self.config)
            train_predictors, validation_predictors, test_predictors = split(predictors, split_lengths)
            train_targets, validation_targets, test_targets = split(targets, split_lengths)

            # construct (predictor_window, target_window) pairs 
            train_predictors, train_targets = extract_observations(train_predictors, train_targets, self.config)
            validation_predictors, validation_targets = extract_observations(validation_predictors, validation_targets, self.config)
            test_predictors, test_targets = extract_observations(test_predictors, test_targets, self.config)

            train.append((train_predictors, train_targets))
            validation.append((validation_predictors, validation_targets))
            test.append((test_predictors, test_targets))

            self.data = dict(
                train=train, 
                validation=validation, 
                test=test
                )
        else: 
            train = [] 
            train_predictors, train_targets = predictors, targets
            train_predictors, train_targets = extract_observations(train_predictors, train_targets, self.config)
            train.append((train_predictors, train_targets))
            self.data = dict(train=train)

    def setup(self): 
        source_path: Path = Path(self.config.data_source)
        source_data: List[Dict[str, np.ndarray]] = load_source(source_path)
        self.log.info(f"Deserialized source data from: {source_path.absolute().as_posix()}")

        train: List[Tuple[np.ndarray]] = [] 
        validation: List[Tuple[np.ndarray]] = [] 
        test: List[Tuple[np.ndarray]] = [] 

        for source in source_data: 
            predictors: np.ndarray = extract_feature_collection(source, self.config.predictors, self.config.standardize, self.config.upsample_factor)
            targets: np.ndarray = extract_feature_collection(source, self.config.targets, self.config.standardize, self.config.upsample_factor)

            # preprocessing
            preprocess(predictors, self.config)
            preprocess(targets, self.config)

            # split the time series
            num_timesteps: int = predictors.shape[-1]
            split_lengths: List[int] = compute_split_lengths(num_timesteps, self.config)
            train_predictors, validation_predictors, test_predictors = split(predictors, split_lengths)
            train_targets, validation_targets, test_targets = split(targets, split_lengths)

            # construct (predictor_window, target_window) pairs 
            train_predictors, train_targets = extract_observations(train_predictors, train_targets, self.config)
            validation_predictors, validation_targets = extract_observations(validation_predictors, validation_targets, self.config)
            test_predictors, test_targets = extract_observations(test_predictors, test_targets, self.config)

            train.append((train_predictors, train_targets))
            validation.append((validation_predictors, validation_targets))
            test.append((test_predictors, test_targets))

        if self.config.aggregate_benchmarks: 
            self.log.info("Aggregating predictors and targets across benchmarks")
            train_predictors: np.ndarray = np.vstack([predictors for predictors, _ in train])
            validation_predictors: np.ndarray = np.vstack([predictors for predictors, _ in validation])
            test_predictors: np.ndarray = np.vstack([predictors for predictors, _ in test])

            train_targets: np.ndarray = np.vstack([targets for _, targets in train])
            validation_targets: np.ndarray = np.vstack([targets for _, targets in validation])
            test_targets: np.ndarray = np.vstack([targets for _, targets in test])

            dataset: Dict[str, np.ndarray] = {
                    "train_predictors": train_predictors, 
                    "train_targets": train_targets, 
                    "validation_predictors": validation_predictors, 
                    "validation_targets": validation_targets, 
                    "test_predictors": test_predictors, 
                    "test_targets": test_targets
                    }

            self.data = dataset 
        else: 
            # TODO: use a more appropriate data structure for this 
            self.data = (train, validation, test)
