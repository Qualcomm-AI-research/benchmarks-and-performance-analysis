# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
from pathlib import Path 
from typing import List, Dict, Tuple, Optional

import numpy as np 
from statsmodels.tsa.tsatools import lagmat 
from tqdm import tqdm

from custom_logging import setup_logger
from io_utils import DATA_DIRECTORY, human_bytes_str, deserialize, serialize

parser = argparse.ArgumentParser(description="""
                                 Script to generate train, validation, and test splits for time series forecasting datasets. 
                                 Each observation is comprised of a tuple (predictor_history, target) where `predictor_history`
                                 is a `p x m` ndarray containing the values of `p` predictors across `m` timesteps and `target` 
                                 is a `k x n` ndarray containing the values of `k` targets with a rollout of `n` timesteps. 
                                 """)

parser.add_argument_group("Paths")
parser.add_argument("--source", type=str, required=True, help="Path to serialized data source to use: should deserialize into a List[Dict[str, np.ndarray]] type.")
parser.add_argument("--output", "-o", type=str, default="dataset.pkl", help="Output path.")

parser.add_argument_group("Predictor/target configuration")
parser.add_argument("--memory", "-m", default=3, type=int, help="Model 'memory': the number of timesteps for which the values of the \
                                                                predictors are given as input to the model.")
parser.add_argument("--predictors", default="ipc", type=str, choices=["ipc", "memory", "operations"], help="Predictor collection to use.")
parser.add_argument("--targets", default="ipc", type=str, choices=["ipc"], help="Target collection to use.")
parser.add_argument("--rollout", "-n", default=1, type=int, help="Number of timesteps to rollout from the model.")

parser.add_argument_group("Preprocessing")
parser.add_argument("--winsorize", action="store_true", help="Clip the front and/or back of the time series.")
parser.add_argument("--front-clip-proportion", type=float, default=0.05, help="Proportion of the front of the time series to be clipped.")
parser.add_argument("--back-clip-proportion", type=float, default=0.05, help="Proportion of the back of the time series to be clipped.")

parser.add_argument_group("Split configuration")
parser.add_argument("--train-proportion", default=0.70, type=float, help="Proportion of the data to use for training (in-sample) fitting.")
parser.add_argument("--validation-proportion", default=0.20, type=float, help="Proportion of the data to use for validation (in-sample) fitting.")
parser.add_argument("--test-proportion", default=0.10, type=float, help="Proportion of the data to use for test (out-of-sample) evaluation.")

def round_to_multiple(x: int, multiple: int) -> int: 
    return x - (x % multiple)

def winsorize(x: np.ndarray, args) -> np.ndarray: 
    n: int = x.shape[-1]
    return x[:, int(n * args.front_clip_proportion):-int(n * args.back_clip_proportion)]

def preprocess(features: np.ndarray, args) -> np.ndarray: 
    if args.winsorize:
        features = winsorize(features, args)

    return features

def extract_feature_collection(data: Dict[str, np.ndarray], collection: Optional[str]="ipc") -> np.ndarray: 
    if collection == "ipc": 
        return data["system.cpu_cluster.cpus.ipc"]
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
        return np.vstack([data[identifier] for identifier in signal_identifiers])
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
        return np.vstack([data[identifier] for identifier in signal_identifiers])
    else: 
        raise NotImplementedError(f"Collection identifier {collection} not supported.")

def compute_split_lengths(num_timesteps: int, args) -> List[int]: 
    assert abs(sum([args.train_proportion, args.validation_proportion, args.test_proportion]) - 1.0) <= 1e-5, f"Provided split configuration does not have proportions that sum to 1.0" 
    num_train: int = int(num_timesteps * args.train_proportion)
    num_validation: int = int((num_timesteps - num_train) * (args.validation_proportion / (args.validation_proportion + args.test_proportion)))
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

def extract_observations(predictors: np.ndarray, targets: np.ndarray, args) -> Tuple[np.ndarray]: 
    num_timesteps: int = predictors.shape[-1] 
    available_timesteps: int = num_timesteps - args.rollout
    n: int = round_to_multiple(available_timesteps, args.memory)

    if targets.ndim == 1: 
        targets = targets[None, :]

    _get_subsequences: callable = lambda x, indices: np.take(x, indices[:, np.newaxis] + np.arange(args.rollout), axis=1).squeeze().T

    predictors: np.ndarray = np.stack(np.split(predictors[:, :n], n // args.memory, axis=-1))
    targets: np.ndarray = _get_subsequences(targets, np.arange(args.memory, num_timesteps, step=args.memory))

    if targets.ndim == 1: 
        targets = targets[:, None]

    return predictors, targets

def main(args): 
    log = setup_logger(__name__) 

    # deserialize the source data 
    source_path: Path = Path(args.source) 
    if not source_path.exists(): 
        raise FileNotFoundError(f"Did not find a file at: {source_path.absolute().as_posix()}.")

    source_data: List[Dict[str, np.ndarray]] = deserialize(source_path.absolute().as_posix())
    log.info(f"Deserialized source data from: {source_path.absolute().as_posix()}")

    train: List[Tuple[np.ndarray]] = [] 
    validation: List[Tuple[np.ndarray]] = [] 
    test: List[Tuple[np.ndarray]] = [] 

    for source in tqdm(source_data): 
        predictors: np.ndarray = extract_feature_collection(source, args.predictors)
        targets: np.ndarray = extract_feature_collection(source, args.targets)

        num_timesteps: int = predictors.shape[-1]

        split_lengths: List[int] = compute_split_lengths(num_timesteps, args)

        # split the time series
        train_predictors, validation_predictors, test_predictors = split(predictors, split_lengths)
        train_targets, validation_targets, test_targets = split(targets, split_lengths)

        # construct (predictor_window, target_window) pairs 
        train_predictors, train_targets = extract_observations(train_predictors, train_targets, args)
        validation_predictors, validation_targets = extract_observations(validation_predictors, validation_targets, args)
        test_predictors, test_targets = extract_observations(test_predictors, test_targets, args)

        train.append((train_predictors, train_targets))
        validation.append((validation_predictors, validation_targets))
        test.append((test_predictors, test_targets))

    # aggregate the predictors/targets across benchmarks 
    log.info("Aggregating predictors and targets across benchmarks")
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
    
    output_path: Path = Path(args.output)
    serialize(dataset, output_path)
    log.info(f"Serialized dataset to: {output_path.absolute().as_posix()}")
    log.info(f"Number of train examples: {train_predictors.shape[0]}")
    log.info(f"Number of validation examples: {validation_predictors.shape[0]}")
    log.info(f"Number of test examples: {test_predictors.shape[0]}")

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
