# Frugal AI Challenge
---


## Data
---
Datasets for this challenge are hosted on [Hugging Face](https://huggingface.co/datasets/rfcx/frugalai) and contain the raw audio arrays, sampling rates, and labels. A class label of 0 means an audio file has a chainsaw in the audio and a label of 1 means there is no chainsaw.

Some exploratory data analysis was done in the `scripts/eda.ipynb` notebook and has some data visualizations embedded within.

There is one script called `scripts/prep_data.py` that is responsible for downloading and preprocessing the audio data. The script will do the following:
1. Download datasets from Hugging Face
2. Remove three "bad samples" (zero-length audio arrays) from the test set that were uncovered during exploratory data analysis (EDA)
3. Remove outliers from training set based on audio durations
4. Downsample the majority class in the train set so that the distribution of class labels are even (initially there are more non-chainsaw audio samples than chainsaw audio samples)
5. Pad train and test set audio arrays with zeros so that they are all the same length
6. Convert train and test audio arrays to mel spectrograms
7. Save arrays to `data/` folder

## Model
--
The model is a simple convolutional neural network written in Jax.

## Folder structure:
* scripts: for ipynbs and one off scripts. 

## Training
---
To train the model:
```bash
python entry_point.py train --config_path=configs/baseline.yaml
```

To run a benchmarking script:
```bash
python scripts/benchmark.py --layer_dims 1 16 32 64 128 --kernel_size 3 --epochs 10 --dtype float32 --batch_size 32
```

## Environment
---
This repo was developed using the following libraries:
* python==3.12.8
* jax==0.4.38
* jaxtyping==0.2.36
* optax==0.2.4
* equinox==0.11.10
* datasets==3.2.0
* einops==0.8.0
* notebook==7.3.1
* numpy==2.0.2
* pyrallis==0.3.1
* scikit-learn==1.6.0

To see and use the full environment refer to `environment.yml`. It is an export of the conda environment used during development of this codebase. If you have conda installed on your machine and want to create create an identical environment with all the libraries ready to go, run this:
```bash
conda env create -f environment.yml
```

Then activate the environment:
```bash
conda activate chainsaw
```

TODO:
- [X] downsample data so classes
- [X] remember to resample audio during inference and on test data
- [X] create script `prep_data.py` that downloads and preprocesses data and saves to numpy arrays in data/
- [X] drop 0-length arrays in test set after downsampling
