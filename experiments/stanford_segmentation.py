# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse 
from pathlib import Path 

import matplotlib.pyplot as plt 

from custom_logging import setup_logger
from dataset import SegmentationConfig, SegmentationDataset
from io_utils import setup_experiment_directory

parser = argparse.ArgumentParser(description="Stanford benchmark segmentation experiment.")
parser.add_argument("--source", type=str, required=True, help="Path to serialized data source.")

def main(args): 
    # setup logging 
    artifact_directory: Path = setup_experiment_directory("stanford_segmentation")
    log_path: Path = artifact_directory / "log.out"
    log = setup_logger(__name__, custom_handle=log_path)

    # load dataset 
    log.info(f"Loading dataset...")
    data_config = SegmentationConfig(
        data_source=Path(args.source), 
    )
    dataset = SegmentationDataset(data_config) 
    log.info(f"Finished loading dataset...")
    log.info(dataset)

    for i, ts in enumerate(dataset.data): 
        plt.figure() 
        plt.title("IPC versus Tick")
        plt.xlabel("Time [us]")
        plt.ylabel("IPC")
        plt.plot(dataset.data[i], c="tab:blue")
        save_path: Path = artifact_directory / f"ts_plot_{i}" 
        plt.savefig(save_path)
        plt.close()

    log.info(f"Saved timeseries plots to: {artifact_directory}")

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
