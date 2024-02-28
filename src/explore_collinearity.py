# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy 
import pathlib 
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np 

from io_utils import DATA_DIRECTORY, deserialize

def summarize_stats(feature_id: str, values: np.ndarray, stats: Dict[str, float]): 
    rms: callable = lambda x: np.linalg.norm(x) / np.sqrt(x.size)
    num_unique_values: int = np.unique(values).size
    stats["num_unique"].append(num_unique_values)

    std: Union[float, str] = values.std() if num_unique_values > 2 else "N/A"

    if std != "N/A": 
        stats["standard_deviation"].append(std)

    stats["minimum_value"].append(values.min())
    stats["maximum_value"].append(values.max())
    stats["dynamic_range"].append(values.max() - values.min())
    stats["average_value"].append(values.mean())
    stats["rms_value"].append(rms(values))
    stats["dirichlet_energy"].append(rms(np.diff(values)))
    stats["value_changes"].append(np.nonzero(np.diff(values))[0].size)
    stats["density"].append((np.nonzero(values)[0].size / values.size) * 100.0)
    nonmedial_density: float = (np.sum(values != np.median(values)) / values.size) * 100.0
    stats["nonmedial_density"].append(nonmedial_density)

    print("\n\n")
    print(f"Feature ID: {feature_id}")
    print("-" * 30)
    print(f"Number of unique values: {num_unique_values}")
    print(f"Minimum value: {values.min():0.5f}")
    print(f"Maximum value: {values.max():0.5f}")
    print(f"Median value: {np.median(values):0.5f}")
    print(f"Dynamic range: {values.max() - values.min():0.5f}")
    print(f"Average value: {values.mean():0.5f}")

    if std == "N/A": 
        print(f"Standard deviation: {std}")
    else: 
        print(f"Standard deviation: {std:0.5f}")

    print(f"Root mean-square value: {rms(values):0.5f}")
    print(f"Dirichlet energy: {rms(np.diff(values)):0.5f}")
    print(f"Number of value changes: {np.nonzero(np.diff(values))[0].size}")
    print(f"Density: {(len(np.nonzero(values)[0]) / values.size) * 100.0:0.5f}%")
    print(f"Non-medial density: {nonmedial_density:0.5f}%")

def custom_filter(results: Dict[str, np.ndarray], filters: Sequence[callable], winsor: Optional[Tuple[int]]=(0, 0)) -> Dict[str, np.ndarray]:
    filtered_results: Dict[str, np.ndarray] = {} 

    for feature_id, values in results.items(): 
        if any([f(values[winsor[0]:-winsor[0]]) for f in filters]): 
            continue
        else: 
            filtered_results[feature_id] = values[winsor[0]:-winsor[0]]

    return filtered_results

def filter_duplicates(results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]: 
    keys: List[str] = list(results.keys())
    features: np.ndarray = np.vstack([x for x in results.values()])

    X: np.ndarray = features / np.linalg.norm(features, axis=1)[:, None]
    out = X @ X.T

    updated_keys: List[str] = copy.deepcopy(keys)
    key_indices_to_remove: set = set()

    for i, key in enumerate(keys): 
        if i in key_indices_to_remove: 
            continue

        duplicate_indices = np.nonzero((1.0 - out[i]) < 1e-2)[0]
        for j in duplicate_indices: 
            if j != i: 
                key_indices_to_remove.add(j)

    updated_results = {} 
    keys_to_remove = [keys[i] for i in key_indices_to_remove]

    for key in keys: 
        if key not in keys_to_remove: 
            updated_results[key] = results[key]

    return updated_results

def low_rank(A, rank):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U = U[:, :rank]  # Select the first 'rank' columns of U
    S = S[:rank]  # Select the first 'rank' singular values
    Vt = Vt[:rank, :]  # Select the first 'rank' rows of Vt
    return U @ np.diag(S) @ Vt

def main(): 
    results_path: Path = Path(DATA_DIRECTORY / "coremark_stats.pkl")
    results: Dict[str, np.ndarray] = deserialize(results_path)

    for key, value in results.items(): 
        results[key] = np.nan_to_num(value, posinf=0.)


    blacklist: List[str] = [
        "system.mem_ctrl.dram.rank1.averagePower", 
        "system.mem_ctrl.dram.rank0.pwrStateTime::REF", 
        "system.mem_ctrl.numStayReadState"
        ]

    IPC: np.ndarray = np.nan_to_num(results["system.cpu.exec_context.thread_0.numInsts"] / results["system.cpu.numCycles"])


    for key in blacklist: 
        del results[key]

    feature_identifiers: List[str] = list(results.keys())
    feature_values: np.ndarray = np.vstack([x for x in results.values()])
    num_features, num_timesteps = feature_values.shape

    print(f"Number of features: {num_features}")
    print(f"Number of timesteps: {num_timesteps}")

    # preprocessing
    static_filter: callable = lambda x: np.nonzero(np.diff(x))[0].size < 5
    nan_filter: callable = lambda x: np.isnan(x).sum() > 0 


    filters: List[callable] = [static_filter, nan_filter]

    winsor_range = (100, 100)
    IPC = IPC[winsor_range[0]:-winsor_range[1]]



    results = custom_filter(results, filters, winsor=winsor_range)
    feature_identifiers: List[str] = list(results.keys())
    feature_values: np.ndarray = np.vstack([x for x in results.values()])
    print(f"Filtered {num_features - len(feature_identifiers)} features for having fewer than 5 changes leaving {len(feature_identifiers)}")

    num_features, num_timesteps = feature_values.shape

    results = filter_duplicates(results)
    feature_identifiers: List[str] = list(results.keys()) + ["IPC"]
    feature_values: np.ndarray = np.vstack([x for x in results.values()] + [IPC])

    print(f"Filtered {num_features - len(feature_identifiers)} features for duplication, leaving {len(feature_identifiers)}")


    stats: Dict[str, list] = dict(
        num_unique=[], 
        minimum_value=[], 
        maximum_value=[], 
        dynamic_range=[], 
        average_value=[], 
        standard_deviation=[], 
        rms_value=[], 
        dirichlet_energy=[], 
        density=[], 
        nonmedial_density=[], 
        value_changes=[]
    )

    for feature_id, values in zip(feature_identifiers, feature_values): 
        summarize_stats(feature_id, values, stats)

    plt.figure() 
    X_norm = feature_values / np.linalg.norm(feature_values, axis=1)[:, None]
    X = np.nan_to_num(np.log(X_norm), neginf=0.)
    plt.imshow((X[:, winsor_range[0]:-winsor_range[1]] != 0.).astype(bool), cmap="Greys", aspect="auto")
    plt.xticks(np.arange(X.shape[1], step=1000))
    plt.yticks([])
    plt.xlabel("Time")
    plt.ylabel("Feature")
    plt.savefig("sparsity_pattern")
    plt.close()

    plt.figure() 
    for name, x in zip(feature_identifiers, X_norm): 
        plt.plot(x, label=name)
    plt.legend()
    plt.savefig("timeseries")
    plt.close()

    plt.figure() 
    order = 25 
    smooth: callable = lambda x, order=order: np.convolve(x, np.ones(order) / order, mode="same")
    plt.plot(feature_values[-1], c="tab:blue", label="IPC", alpha=0.5)
    plt.plot(smooth(feature_values[-1])[order:-order], c="tab:blue", label=f"IPC (Low-pass filtered)")
    plt.legend()
    plt.savefig("ipc")
    plt.close()

    
    plt.figure()
    _, r = np.linalg.qr(X)
    plt.imshow(np.clip(np.abs(r[:, :100]), 0., 0.1), cmap="Greys")
    plt.xticks([])
    plt.yticks(np.arange(X.shape[0]))
    plt.savefig("qr")
    plt.close()


    colors: List[str] = [f"tab:{color}" for color in ["blue", "red", "green", "orange", "purple", "brown"]]

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True)
    order = 5

    for i, ax in enumerate(axs): 
        try: 
            n: int = 100
            idx = i*2
            ax[0].plot(feature_values[idx, :n], c=colors[idx], label=feature_identifiers[idx], alpha=0.5)
            smoothed = smooth(feature_values[idx, :n], order=order)
            ax[0].plot(smoothed[order*2:-order*2], c=colors[idx], label=f"{feature_identifiers[idx]} (Low-pass filtered)")
            ax[0].legend(loc="upper right")

            ax[1].plot(feature_values[idx + 1, :n], c=colors[idx + 1], label=feature_identifiers[idx + 1], alpha=0.5)
            ax[1].plot(smooth(feature_values[idx+1, :n], order=order)[order*2:-order*2], c=colors[idx+1], label=f"{feature_identifiers[idx+1]} (Low-pass filtered)")
            ax[1].legend(loc="upper right")
        except: 
            continue

    fig.set_size_inches(18.5, 10.5)


    plt.savefig("features")
    plt.close()

    

    breakpoint()
    dummy: int = 5


if __name__=="__main__": 
    main()
