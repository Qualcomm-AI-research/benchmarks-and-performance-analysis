# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
from pathlib import Path 
from typing import Dict, List, Set

import numpy as np 

from custom_logging import setup_logger
from io_utils import deserialize, serialize

parser = argparse.ArgumentParser(description="Aggregates and serializes a collection of simulation telemetry results.")
parser.add_argument("--results", type=str, required=True, help="Path to directory containing simulation results.")
parser.add_argument("--extension", type=str, default=".custom.pkl", help="File extension for serialized telemetry results.")
parser.add_argument("--output", "-o", type=str, default="aggregated.pkl", help="Path to save the output.")

def main(args): 
    log = setup_logger(__name__) 

    # collect list of paths to telemetry outputs by recursively searching the results directory
    results_dir: Path = Path(args.results) 
    log.info(f"Aggregating results from: {results_dir.as_posix()}")

    telemetry_paths: List[Path] = list(results_dir.rglob(f"*{args.extension}"))

    if not len(telemetry_paths): 
        raise FileNotFoundError(f"Could not find any results with extension {args.extension} within directory: {args.results}")

    # collect results
    log.info(f"Collecting results from {len(telemetry_paths)} simulations...")
    results: Dict[Path, np.ndarray] = {path: deserialize(path) for path in telemetry_paths}
    log.info("Finished collecting results...")

    timeseries: List[Dict[str, np.ndarray]] = list(results.values())
    feature_sets: List[Set[str]] = [set(list(x.keys())) for x in timeseries]
    num_features: List[int] = [len(feature_set) for feature_set in feature_sets] # number of distinct features/variables collected for each benchmark


    if not all(f == feature_sets[0] for f in feature_sets): 
        log.info(f"Provided benchmarks collect a variable collection of features:")
        log.info(f"\tMinimum: {min(num_features)}")
        log.info(f"\tMaximum: {max(num_features)}")
        log.info(f"\tAverage: {sum(num_features) / len(num_features)}")

        log.info("Taking the intersection of features across all benchmarks...")
        features: Set[str] = feature_sets[0].intersection(*feature_sets[1:])

        for ts in timeseries: 
            for feature in list(ts.keys()): 
                if feature not in features: 
                    ts.pop(feature)
    else: 
        features: Set[str] = feature_sets[0]


    log.info(f"\tNumber of features: {len(features)}")

    # save 
    output_path: Path = Path(args.output)
    serialize(timeseries, output_path)
    log.info(f"Serialized results and saved to: {output_path.as_posix()}")

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
