# rnn-classifier

## Introduction

This recurrent neural network (RNN) deep learning model uses supervised learning to classify time-series/sequential data into predefined separate classes. It was initially designed to classify different classes of astronomical objects observed by the Gravitational-wave Optical Transient Observer (GOTO) survey telescope.

This model was designed to handle class imbalance in supervised classification, without the need for data augmentation methods. The classifier uses weighted loss functions (cross entropy and focal loss) to account for class imbalance and handle classifying rare events in the data.

The model is also able to aggregate different types of data, with the option of feeding the model a sequential/time-series component and additional ‘metadata’ (additional features that are not necessarily time/sequence dependent). The classifier is able to learn deep representations from the mixed data type input to provide classifications.

## Applications
The use of this classifier on data from the GOTO survey is reported in [Burhanudin et al. 2020](https://arxiv.org/pdf/2105.11169.pdf). Below are some results taken from the paper: a confusion matrix showing performance on classifying objects as either a variable star (VS), supernova (SN) or active galactic nuclei (AGN), and the area under the receiver operating characteristic curve (AUC) score evaluated with an increasing number of observations. Note the difference in the number of samples for variable stars (~30,000), supernovae (~60), and active galactic nuclei (~160).
<p align="center">
  <img src="https://github.com/RSE-Sheffield/rnn_classifier/blob/main/images/goto_results.png" width="860"/>
</p>

This classifier was also used on the ECG Heartbeat Categorization Dataset from [Kaggle](https://www.kaggle.com/shayanfazeli/heartbeat), to illustrate how this can be applied to datasets outside of astronomy.
<p align="center">
  <img src="https://github.com/RSE-Sheffield/rnn_classifier/blob/main/images/ecg_results.png" width="860"/>
</p>

## Model architecture

The classifier uses a RNN to extract features from sequential/time-series data, which is then passed to a standard dense neural network where the feature-class mapping is learnt. The additional metadata is combined with features extracted from the RNN component before passing through the dense neural network.

## Building and running the Dockerfile

Buid a docker and run it 

```
# cd to the rnn_classifier's directory 
cd rnn_classifier
docker build . --tag rnn_classifier
docker run -it rnn_classifier
```

## Preprocessing data

The model requires a training, validation, and test set to begin training. The labels need to be in a [one-hot encoded](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) format.

The labels should be saved as `.npy` files, name `x_labels.npy` where `x` is either `train`, `validation`, or `test`.

The time-series features need to be formatted so that the input shape for each examples is `(steps, features)` where `steps` represents the number time-steps, and `features` represents the number of features at each time step. For sequences that have different lenghts, [masking](https://www.tensorflow.org/guide/keras/masking_and_padding) should be used - the mask value can be set in the config `.yaml` file. Time-series features should be saved as `.npy` files, named `x_time_features.npy` where `x` is either `train`, `validation`, or `test`.

Metadata features need to be in the format `(features)` where `features` represents the number of features. Metadta features should be saved as `.npy` files, named `x_contextual_features.npy` where `x` is either `train`, `validation`, or `test`.

## Training models

An example dataset is included in `data/sample_data`. The data contains examples of simulated time-series data with errors (noise), separated into two classes (class labels 0 and 1).

Example trained models are included in `trained_models`.

To train a grid of models with the example data, run:

```
python3 mixed_input_train.py -d sample_data -c mixed_input_train_config.yaml
```

This will train a grid of models, and save them to a folder called `trained_models`, where each grid of trained models will be saved into a folder named according to the time that it was trained, the type of recurrent neural network, and the loss function used. For example: `2021-8-5-1103-gru-fl` contains models trained on 5th August 2021 at 11:03 am using the GRU architecture with the focal loss function.

Within each configuration folder, there will be folders named according to the model hyperparameters e.g. `32_0.2_25_1.0_0.0001_0.01_50` which contain the trained models with those hyperparameters. There is also a `.csv` file named `training_results.csv` tabulating the F1 score and accuracy for models trained with each set of hyperparameters.

To train models without including additional metadata, and just use sequential/time-series data, add `--mixed False` or `-mi False`. The default value is set to True, where the model will ingest both sequential/time-series data and additional metadata.


## Evaluating trained models

To evaluate a grid trained models, run:

```
python3 model_evaluation.py -d sample_data -m trained_models/2021-9-20-1037-gru-fl -l "Class 0" -l "Class 1"
```

The `trained_models/2021-9-20-1037-gru-fl` is the folder where the trained model is saved.
"Class 0" and "Class 1" are the class labels (for a binary classification case). 

This will evaluate the models, and tabulate the model hyperparameters with F1 score and accuracy into `experiment_results.csv` within the configuration folder. It will also create two confusion matrices (evaluated on the validation and test set).

For models trained without additional metadata, add the `--mixed False` or `-mi False` flags as when training the models.

## References
Burhanudin et al.,
*Light-curve classification with recurrent neural networks for GOTO: dealing with imbalanced data*,
Monthly Notices of the Royal Astronomical Society, Volume 505, Issue 3, August 2021,
Pages 4345–4361, https://doi.org/10.1093/mnras/stab1545
