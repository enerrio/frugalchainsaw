# Frugal AI Challenge
---


## Data
---
Datasets for this challenge are hosted on [Hugging Face](https://huggingface.co/datasets/rfcx/frugalai) and contain the raw audio arrays, sampling rates, and labels. A class label of 0 means an audio file has a chainsaw in the audio and a label of 1 means there is no chainsaw.

Some exploratory data analysis was done in the `scripts/eda.ipynb` notebook and has some data visualizations embedded within.

There is one script called `scripts/prep_data.py` that is responsible for downloading and preprocessing the audio data. The script will do the following:
1. Download datasets from Hugging Face
2. Remove three "bad samples" (zero-length audio arrays) from the test set that were discovered during exploratory data analysis
3. Remove outliers from training set based on audio durations
4. Downsample the majority class in the train set so that the distribution of class labels are even (initially there are more non-chainsaw audio samples than chainsaw audio samples)
5. Pad train and test set audio arrays with zeros so that they are all the same length
6. Convert train and test audio arrays to mel spectrograms
7. Normalize the data either globally or per frequency bin
8. Save arrays to `data/` folder

## Environment
---
Dependencies are managed using [uv](https://docs.astral.sh/uv/). To install dependencies, run:
```bash
uv sync
```

Check out `pyproject.toml` to see all the dependencies that are required. The main requirements are:
* python==3.12.8
* jax
* jaxtyping
* optax
* equinox
* datasets
* einops
* pyrallis
* scikit-learn


## Model
--
The model is a simple convolutional neural network written in Jax. Batch normalization is applied to each layer and a leaky ReLU activation function is applied after each layer. The final layer is a fully connected layer that outputs a single value for each sample.

The model's weights are initialized using a custom initialization scheme defined in the `reinit_model_params` function. This function also allows the user to specify the data type for the model's weights.

During training the model uses `sigmoid_binary_cross_entropy` loss function from [Optax](https://optax.readthedocs.io/en/latest/) which expects the model's output to be a single value for each sample i.e. logits.

## Folder structure:
* scripts: for ipynbs and one off scripts. 
* data: Preprocessed audio data stored as numpy arrays
* configs: model configuration files
* src: source code for the model and training loop
* tests: unit tests

## Training
---
To train the model:
```bash
uv run entry_point.py train --config_path=configs/baseline.yaml
```

To plot results:
```bash
uv run entry_point.py plot --config_path=configs/baseline.yaml
```

To evaluate on test set:
```bash
uv run entry_point.py eval --config_path=configs/baseline.yaml
```

To run a benchmarking script:
```bash
uv run entry_point.py benchmark --config_path=configs/baseline.yaml
```

To download all dependencies and preprocess data on a new machine:
```bash
chmod +x setup.sh
./setup.sh git@github.com:enerrio/frugalchainsaw.git main
```

## Post-training
---
Following training, we can optimize the model for deployment in a number of ways. One way is to quantize the model.


TODO:
- [X] downsample data so classes
- [X] remember to resample audio during inference and on test data
- [X] create script `prep_data.py` that downloads and preprocesses data and saves to numpy arrays in data/
- [X] drop 0-length arrays in test set after downsampling
- [X] try applying normalization per frequency bin
- [X] fix estimated train time in benchmark script
  - [X] 25/33 minutes for training. benchmark says 7 though
- [X] fix progress bar name. should say batches, not epochs
- [X] write evaluate script to evaluate model on test set
- [X] evaluate script: add confusion matrix plot
- [ ] use code carbon to estimate hardware energy consumption
- [ ] train on float16 baseline (10 epochs)
- [ ] train on bfloat16 baseline (10 epochs)
- [ ] train on deep baseline (10 epochs)
- [ ] train on medium baseline w/ weight decay (50 epochs)
- [ ] train on longer baseline (100 epochs)
- [ ] read those papers
- [ ] model optimization in tensorflow link
- [ ] quantize model and measure on test set
- [X] replace conda with uv
