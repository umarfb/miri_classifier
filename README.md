# rnn-classifier

## Introduction

This recurrent neural network (RNN) deep learning model uses supervised learning to classify time-series/sequential data into predefined separate classes. It was initially designed to classify different classes of astronomical objects observed by the Gravitational-wave Optical Transient Observer (GOTO) survey telescope.

This model was designed to handle class imbalance in supervised classification, without the need for data augmentation methods. The classifier uses weighted loss functions (cross entropy and focal loss) to account for class imbalance and handle classifying rare events in the data.

The model is also able to aggregate different types of data, with the option of feeding the model a sequential/time-series component and additional ‘metadata’ (additional features that are not necessarily time/sequence dependent). The classifier is able to learn deep representations from the mixed data type input to provide classifications.

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

## Training models

An example dataset is included in `data/sample_data`. The data contains examples of simulated time-series data with errors (noise), separated into two classes (class labels 0 and 1).

Example trained models are included in `trained_models`.

To train a grid of models with the example data, run:

```
python3 mixed_input_train.py -d sample_data -c mixed_input_train_config.yaml
```

This will train a grid of models, and save them to a folder called `trained_models`, where each grid of trained models will be saved into a folder named according to the time that it was trained, the type of recurrent neural network, and the loss function used. For example: `2021-8-5-1103-gru-fl` contains models trained on 5th August 2021 at 11:03 am using the GRU architecture with the focal loss function.

Within each configuration folder, there will be folders named according to the model hyperparameters e.g. `32_0.2_25_1.0_0.0001_0.01_50` which contain the trained models with those hyperparameters. There is also a `.csv` file named `training_results.csv` tabulating the F1 score and accuracy for models trained with each set of hyperparameters.

## Evaluating trained models

To evaluate a grid trained models, run:

```
python3 model_evaluation.py -d sample_data -m trained_models/2021-9-20-1037-gru-fl -l "Class 0" -l "Class 1"
```

The `trained_models/2021-9-20-1037-gru-fl` is the folder where the trained model is saved.
"Class 0" and "Class 1" are the class labels (for a binary classification case). 

This will evaluate the models, and tabulate the model hyperparameters with F1 score and accuracy into `experiment_results.csv` within the configuration folder. It will also create two confusion matrices (evaluated on the validation and test set). 
