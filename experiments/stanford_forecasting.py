# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse 
from pathlib import Path 

import matplotlib.pyplot as plt 
import jax.numpy as np 
from jax import jit, grad, vmap 
import jax.nn as nn 

from custom_logging import setup_logger
from dataset import ForecastingConfig, ForecastingDataset
from io_utils import EXPERIMENT_LOGS_DIRECTORY
from metrics import rms, percentage_error
from models import last_value, running_average

parser = argparse.ArgumentParser(description="Stanford benchmark forecasting experiment.")

parser.add_argument("--source", type=str, required=True, help="Path to serialized data source.")
parser.add_argument("--model", type=str, default="baseline", choices=["baseline"], help="Model type to use for fitting the data.")

def report_result(targets: np.ndarray, predictions: np.ndarray, log): 
    log.info(f"Target RMS: {rms(targets):0.3f}")
    log.info(f"RMS Error: {rms(predictions - targets):0.3f}")
    log.info(f"Percentage Error: {percentage_error(predictions, targets):0.3f}%")

def main(args): 
    log = setup_logger(__name__)

    # load dataset 
    log.info(f"Loading dataset...")
    data_config = ForecastingConfig(
        data_source=Path(args.source), 
        aggregate_benchmarks=True
    )
    dataset = ForecastingDataset(data_config) 
    log.info(f"Finished loading dataset...")

    # configure model 
    log.info("Configuring model...")
    if args.model == "baseline":
        log.info(f"using model type: {args.model}")
    else: 
        raise NotImplementedError
    log.info("Finished configuring model...")

    # fit baselines
    validation_predictions: np.ndarray = vmap(last_value)(dataset.data["validation_predictors"])
    report_result(dataset.data['validation_targets'], validation_predictions, log)
    # log.info(f"Model: last value predictor")
    # log.info(f"Target RMS: {rms(dataset.data['validation_targets']):0.3f}")
    # log.info(f"RMS Error: {rms(validation_predictions - dataset.data['validation_targets']):0.3f}")
    # log.info(f"Percentage Error: {percentage_error(validation_predictions, dataset.data['validation_targets']):0.3f}%")

    validation_predictions = [] 
    params = {"sum": 0.0, "count": 1.0}

    for x in dataset.data["validation_predictors"]: 
        params, prediction = running_average(params, x)
        validation_predictions.append(prediction)

    validation_predictions = np.array(validation_predictions)
    log.info(f"Evaluating running average predictor")
    report_result(dataset.data['validation_targets'], validation_predictions, log)


    # log.info(f"Target RMS: {rms(dataset.data['validation_targets']):0.3f}")
    # log.info(f"RMS Error: {rms(validation_predictions - dataset.data['validation_targets']):0.3f}")
    # log.info(f"Percentage Error: {percentage_error(validation_predictions, dataset.data['validation_targets']):0.3f}%")

    log.info("Completed baselines...")

    # fit linear AR
    A: np.ndarray = dataset.data['train_predictors']
    b: np.ndarray = dataset.data['train_targets']
    log.info("Fitting linear AR model")
    x: np.ndarray = np.linalg.solve(A.T @ A, A.T @ b)
    log.info("Finished fitting linear AR model")

    # evaluate linear AR 
    log.info("Evaluating linear AR model")
    validation_predictions: np.ndarray = dataset.data['validation_predictors'] @ x 
    report_result(dataset.data['validation_targets'], validation_predictions, log)
    log.info("Finished evaluating linear AR model")

    # evaluate single layer MLP 
    def mlp(params, x: jnp.ndarray) -> jnp.ndarray: 
        activation = nn.relu(params[0]["weights"] @ x + params[0]["bias"])
        for weight, bias in params[1:-1]: 
            activation = nn.relu(weight @ activation + bias)

        out = params[-1]["weight"] @ activation + params[-1]["bias"]
        return out

    def init_mlp(layer_sizes, seed: int=0, scale: float=1e-1, use_bias): 
        key = jnpr.PRNGKey(seed)
        params = [] 
        for i, j in zip(layer_sizes[:-1], params[1:]): 
            key, current_key = jnpr.split(key)
            params.append(scale * jnpr.randn(i, j))
        pass

        



if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
