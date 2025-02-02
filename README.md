# Frugal AI Challenge
---
The [Frugal AI Challenge](https://frugalaichallenge.org) is an event made up of three machine learning tasks across different modalities with the purpose of encouraging people to keep efficiency in mind when deploying ML models. This repo tackles the `Detecting illegal deforestation` task which is a binary classification task using audio data. The main training and evaluation code is here but for the purposes of the challenge, a scaled down version of the inference code lives in a [HuggingFace Spaces repo](https://huggingface.co/spaces/enerrio/submission-template/tree/main) which handles measuring inference efficiency and task submission.


## Data
---
Datasets for this challenge are hosted on [Hugging Face](https://huggingface.co/datasets/rfcx/frugalai) and contain the raw audio arrays, sampling rates, and labels. A class label of 0 means an audio file has a chainsaw in the audio and a label of 1 means there is no chainsaw.

Some exploratory data analysis was done in the `scripts/eda.ipynb` notebook and has some data visualizations embedded within.

There is one script called `scripts/prep_data.py` that is responsible for downloading and preprocessing the audio data. The script will do the following:
1. Download datasets from Hugging Face
2. Remove three "bad samples" (zero-length audio arrays) from the test set that were discovered during exploratory data analysis (these are not removed from the test set during challenge submission)
3. Remove outliers from training set based on audio durations
4. Downsample the majority class in the train set so that the distribution of class labels are even (initially there are more non-chainsaw audio samples than chainsaw audio samples)
5. Pad train and test set audio arrays with zeros so that they are all the same length
6. Convert train and test audio arrays to mel spectrograms
7. Normalize the data either globally or per frequency bin
8. Save arrays to disk

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
* scripts: for notebooks and one off scripts. 
* data: Preprocessed audio data stored as numpy arrays using global normalization
* data_binwise: Preprocessed audio data stored as numpy arrays using bin-wise normalization
* configs: Model configuration files
* results: Results from training runs
* src: source code for the model and training loop
* tests: unit tests

In addition to the above there are four main entry scripts:
* `run_train.py`: Train a model
* `run_plot.py`: Plot the results of a training run
* `run_eval.py`: Evaluate the model on the test set
* `run_infer.py`: Run inference on the test set and save predictions to disk

Finally, there is also `entry_point.py` that is used as a single point of entry for the all the above scripts in order to enforce `jaxtyping`.

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
- [X] train on float16 baseline (10 epochs)
- [X] train on bfloat16 baseline (10 epochs)
- [X] train on deep baseline (10 epochs)
- [ ] train on medium baseline w/ weight decay (50 epochs)
- [ ] train on longer baseline (100 epochs)
- [ ] quantize model and measure efficiency gains
- [ ] try training on hardware accelerators (GPU/TPU) via Google Colab
- [X] replace conda with uv
