import numpy as np
import json
from datasets import load_dataset
import matplotlib.pyplot as plt

def calculate_mean_std():
    ds = load_dataset("JasiekKaczmarczyk/maestro-sustain-quantized", split="train")

    #  2 ** -5 is used to prevent log2 of 0
    dstart = np.nan_to_num(ds["dstart"]) + (2 ** -5)
    duration = ds["duration"]

    log2_dstart = np.log2(dstart)
    log2_duration = np.log2(duration)

    mean_dstart = np.mean(log2_dstart)
    std_dstart = np.std(log2_dstart)

    mean_duration = np.mean(log2_duration)
    std_duration = np.std(log2_duration)

    features = {
        "mean_dstart": mean_dstart,
        "std_dstart": std_dstart,
        "mean_duration": mean_duration,
        "std_duration": std_duration
    }

    with open("artifacts/time_features.json", "x") as f:
        json.dump(features, f)
        f.close()

def normalize_time_features(time_feature: np.ndarray, mean: float, std: float):
    log2_x = np.log2(time_feature)

    return (log2_x - mean) / std

def compare_unnormalized_and_normalized_time(idx: int):
    ds = load_dataset("JasiekKaczmarczyk/maestro-sustain-quantized", split="train")

    record = ds[idx]

    time_features = json.load(open("artifacts/time_features.json"))

    dstart = np.array(record["dstart"]) + (2 ** -5)
    duration = np.array(record["duration"])
    dstart_norm = normalize_time_features(dstart, time_features["mean_dstart"], time_features["std_dstart"])
    duration_norm = normalize_time_features(duration, time_features["mean_duration"], time_features["std_duration"])

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0][0].hist(dstart)
    ax[0][0].set_title("dstart")

    ax[0][1].hist(dstart_norm)
    ax[0][1].set_title("normalized dstart")

    ax[1][0].hist(duration)
    ax[1][0].set_title("duration")

    ax[1][1].hist(duration_norm)
    ax[1][1].set_title("normalized duration")

    plt.show()
