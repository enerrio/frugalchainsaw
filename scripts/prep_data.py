import os
import random
import argparse
import multiprocessing as mp
from collections import defaultdict
import numpy as np
from rich import print
import librosa
from datasets import load_dataset, Audio, Dataset
from rich.progress import track
from scipy.stats import zscore
from sklearn.model_selection import train_test_split


BAD_TEST_PATHS = {
    "mashpi_2020_8d4d84bd-3b24-4af1-82e2-168b678774f3_64-67.wav",
    "warsi_2021_c6aa9986-56fb-456c-8fef-d4b44622e8f2_82-85.wav",
    "tembe_2018_e5e7f4e2-2dd8-4793-8504-27f346a8d58f_84-87.wav",
}
THREE_SECOND_SAMPLE_LENGTH = 12000 * 3


def calculate_duration(sample: dict) -> float:
    """Calculate audio duration in seconds."""
    audio_length = len(sample["audio"]["array"])
    return audio_length / sample["audio"]["sampling_rate"]


def filter_duration_outliers(dataset: Dataset) -> Dataset:
    """Remove samples with outlier durations."""
    # Calculate durations
    durations = []
    paths = []
    for sample in track(dataset, description="Calculating durations..."):
        duration = calculate_duration(sample)
        durations.append(duration)
        paths.append(sample["audio"]["path"])

    # Find duration outliers
    z = zscore(durations)
    outlier_idxs = np.abs(z) > 3
    bad_sample_paths = {x for i, x in enumerate(paths) if outlier_idxs[i]}
    print(f"Number of duration outliers: {len(bad_sample_paths):,}")

    # Filter outliers
    filtered_dataset = dataset.filter(
        lambda x: x["audio"]["path"] not in bad_sample_paths
    )
    return filtered_dataset


def downsample_majority_class(dataset: Dataset) -> Dataset:
    """Balance dataset by downsampling majority class."""
    # Group samples by label
    label2paths = defaultdict(list)
    for example in track(dataset, description="Grouping samples by label..."):
        label2paths[example["label"]].append(example["audio"]["path"])

    # Get min count
    min_count = min(len(paths) for paths in label2paths.values())
    print(f"Number of samples in minority class: {min_count:,}")

    # Random sample for each class
    balanced_paths = set()
    for paths in label2paths.values():
        balanced_paths.update(random.sample(paths, min_count))

    # Create balanced dataset
    balanced_dataset = dataset.filter(lambda x: x["audio"]["path"] in balanced_paths)

    return balanced_dataset


def pad_and_compute_mel(sample: dict) -> dict:
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    # Trim if audio is longer than target; otherwise, pad.
    if len(audio) > THREE_SECOND_SAMPLE_LENGTH:
        audio = audio[:THREE_SECOND_SAMPLE_LENGTH]
    else:
        pad_width = THREE_SECOND_SAMPLE_LENGTH - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant", constant_values=0)

    # Compute mel spectrogram directly from audio.
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    # Convert power spectrogram to dB scale.
    log_mel_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)
    return {"array": log_mel_spectrogram, "label": sample["label"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalization_mode",
        type=str,
        default="global",
        choices=["global", "binwise"],
    )
    args = parser.parse_args()
    cpu_count = mp.cpu_count()
    print(f"Number of CPUs: {cpu_count}")

    # Load dataset and resample to 12kHz
    print("Loading and resampling dataset...")
    dataset_train = load_dataset("rfcx/frugalai", split="train")
    dataset_test = load_dataset("rfcx/frugalai", split="test")
    dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=12000))
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=12000))
    print(f"Number of training samples: {len(dataset_train):,}")
    print(f"Number of testing samples: {len(dataset_test):,}")
    # Remove bad samples from test set
    dataset_test = dataset_test.filter(
        lambda x: x["audio"]["path"] not in BAD_TEST_PATHS
    )
    print(
        f"Number of testing samples after removing bad samples: {len(dataset_test):,}"
    )

    # Filter duration outliers
    print("Filtering duration outliers...")
    dataset_train = filter_duration_outliers(dataset_train)
    print(
        f"Number of training samples after filtering duration outliers: {len(dataset_train):,}"
    )

    # Balance classes
    print("Balancing classes in training set...")
    dataset_train = downsample_majority_class(dataset_train)
    print(f"Number of training samples after downsampling: {len(dataset_train):,}")

    # Pad audio to three seconds and convert to mel-spectrogram
    print("Processing audio (trimming/padding + computing mel spectrograms)...")
    dataset_train = dataset_train.map(pad_and_compute_mel, num_proc=cpu_count)
    dataset_test = dataset_test.map(pad_and_compute_mel, num_proc=cpu_count)

    # Convert to arrays and save
    print("Saving processed data...")
    X_train, X_test = [], []
    y_train, y_test = [], []

    for sample in track(dataset_train, description="Processing training data..."):
        X_train.append(np.array(sample["array"]))
        y_train.append(sample["label"])
    for sample in track(dataset_test, description="Processing testing data..."):
        X_test.append(np.array(sample["array"]))
        y_test.append(sample["label"])
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,  # Maintain class distribution
    )

    # Normalize data to zero mean and unit variance
    print(f"Normalizing data with mode: {args.normalization_mode}")
    if args.normalization_mode == "global":
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-6  # Avoid division by zero
    elif args.normalization_mode == "binwise":
        # Normalize data per frequency bin
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    else:
        raise ValueError(f"Invalid normalization mode: {args.normalization_mode}")

    # Apply normalization
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print(
        f"X_train shape: {X_train.shape} | mean: {np.mean(X_train):.3f} | std: {np.std(X_train):.3f}"
    )
    print(
        f"y_train shape: {y_train.shape} | class distribution: {np.bincount(y_train)}"
    )
    print(
        f"X_val shape: {X_val.shape} | mean: {np.mean(X_val):.3f} | std: {np.std(X_val):.3f}"
    )
    print(f"y_val shape: {y_val.shape} | class distribution: {np.bincount(y_val)}")
    print(
        f"X_test shape: {X_test.shape} | mean: {np.mean(X_test):.3f} | std: {np.std(X_test):.3f}"
    )
    print(f"y_test shape: {y_test.shape} | class distribution: {np.bincount(y_test)}")

    data_dir = "data" if args.normalization_mode == "global" else "data_binwise"
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    # Save mean and std for inference
    np.save(os.path.join(data_dir, "mean.npy"), mean)
    np.save(os.path.join(data_dir, "std.npy"), std)

    print(f"Done! Saved processed arrays to {data_dir}/")


if __name__ == "__main__":
    main()
