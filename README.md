# rnn-classifier

A Recurrent Neural Network (RNN) classifier designed to classify astronomical time-series data.

An example dataset is included in `data/sample_data`. The data contains examples of simulated time-series data with errors (noise), separated into two classes (class labels 0 and 1).

Example trained models are included in `trained_models`.


## Training models

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
