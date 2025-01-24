import os
import random
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
    duration = audio_length / sample["audio"]["sampling_rate"]
    return duration


def filter_duration_outliers(dataset: Dataset) -> Dataset:
    """Remove samples with outlier durations."""
    # Calculate durations
    durations = []
    paths = []
    for sample in track(
        dataset,
        description="Calculating durations...",
    ):
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
    for example in track(
        dataset,
        description="Grouping samples by label...",
    ):
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


def compute_log_mel_spectrogram(audio: np.ndarray, sr: int = 12000) -> np.ndarray:
    """Compute mel spectrogram from audio."""
    audio_stft = librosa.stft(audio)
    audio_stft_mag, _ = librosa.magphase(audio_stft)
    mel_spectrogram = librosa.feature.melspectrogram(S=audio_stft_mag, sr=sr)
    log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram


def main():
    # Load dataset and resample to 12kHz
    print("Loading and resampling dataset...")
    dataset_train = load_dataset("rfcx/frugalai", split="train")
    dataset_test = load_dataset("rfcx/frugalai", split="test")
    dataset_train_resampled = dataset_train.cast_column(
        "audio", Audio(sampling_rate=12000)
    )
    dataset_test_resampled = dataset_test.cast_column(
        "audio", Audio(sampling_rate=12000)
    )
    print(f"Number of training samples: {len(dataset_train):,}")
    print(f"Number of testing samples: {len(dataset_test):,}")
    # Remove bad samples from test set
    dataset_test_resampled = dataset_test_resampled.filter(
        lambda x: x["audio"]["path"] not in BAD_TEST_PATHS
    )
    print(
        f"Number of testing samples after removing bad samples: {len(dataset_test_resampled):,}"
    )

    # Filter duration outliers
    print("Filtering duration outliers...")
    filtered_train_dataset = filter_duration_outliers(dataset_train_resampled)
    print(
        f"Number of training samples after filtering duration outliers: {len(filtered_train_dataset):,}"
    )

    # Balance classes
    print("Balancing classes in training set...")
    balanced_train_dataset = downsample_majority_class(filtered_train_dataset)
    print(
        f"Number of training samples after downsampling: {len(balanced_train_dataset):,}"
    )

    # Pad audio to three seconds
    print("Padding audio to three seconds...")
    padded_train_dataset = balanced_train_dataset.map(
        lambda x: {
            "audio": {
                "array": np.pad(
                    x["audio"]["array"],
                    (0, THREE_SECOND_SAMPLE_LENGTH - x["audio"]["array"].shape[0]),
                    mode="constant",
                    constant_values=0,
                ),
                "sampling_rate": x["audio"]["sampling_rate"],
            },
            "label": x["label"],
        }
    )
    padded_test_dataset = dataset_test_resampled.map(
        lambda x: {
            "audio": {
                "array": np.pad(
                    x["audio"]["array"],
                    (0, THREE_SECOND_SAMPLE_LENGTH - x["audio"]["array"].shape[0]),
                    mode="constant",
                    constant_values=0,
                ),
                "sampling_rate": x["audio"]["sampling_rate"],
            },
            "label": x["label"],
        }
    )
    print(f"Number of training samples after padding: {len(padded_train_dataset):,}")
    print(f"Number of testing samples after padding: {len(padded_test_dataset):,}")

    # Convert to mel spectograms
    print("Computing mel spectrograms...")
    sgram_train_dataset = padded_train_dataset.map(
        lambda x: {
            "array": compute_log_mel_spectrogram(x["audio"]["array"]),
            "label": x["label"],
        }
    )
    sgram_test_dataset = padded_test_dataset.map(
        lambda x: {
            "array": compute_log_mel_spectrogram(x["audio"]["array"]),
            "label": x["label"],
        }
    )

    # Convert to arrays and save
    print("Saving processed data...")
    X_train, X_test = [], []
    y_train, y_test = [], []

    for sample in track(sgram_train_dataset, description="Processing training data..."):
        X_train.append(np.array(sample["array"]))
        y_train.append(sample["label"])
    for sample in track(sgram_test_dataset, description="Processing testing data..."):
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
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-6  # Avoid division by zero
    # Normalize data per frequency bin
    # mean = X_train.mean(axis=(0, 2), keepdims=True)
    # std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6

    # Apply normalization
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print(f"X_train shape: {X_train.shape} | mean: {np.mean(X_train):.3f} | std: {np.std(X_train):.3f}")
    print(f"y_train shape: {y_train.shape} | class distribution: {np.bincount(y_train)}")
    print(f"X_val shape: {X_val.shape} | mean: {np.mean(X_val):.3f} | std: {np.std(X_val):.3f}")
    print(f"y_val shape: {y_val.shape} | class distribution: {np.bincount(y_val)}")
    print(f"X_test shape: {X_test.shape} | mean: {np.mean(X_test):.3f} | std: {np.std(X_test):.3f}")
    print(f"y_test shape: {y_test.shape} | class distribution: {np.bincount(y_test)}")

    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_val.npy", X_val)
    np.save("data/y_val.npy", y_val)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    print("Done! Saved processed arrays to data/")


if __name__ == "__main__":
    main()
