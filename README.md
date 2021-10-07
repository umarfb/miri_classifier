# rnn-classifier

A Recurrent Neural Network (RNN) classifier designed to classify astronomical time-series data.

An example dataset is included in `data/sample_data`. The data contains examples of simulated time-series data with errors (noise), separated into two classes (class labels 0 and 1).

Example trained models are included in `trained_models`.


## Training models

To train a grid of models with the example data, run:

```
python3 mixed_input_train.py sample_data
```

This will train a grid of models, and save them to a folder called `trained_models`, where each grid of trained models will be saved into a folder named according to the time that it was trained, the type of recurrent neural network, and the loss function used. For example: `2021-8-5-1103-gru-fl` contains models trained on 5th August 2021 at 11:03 am using the GRU architecture with the focal loss function.

Within each configuration folder, there will be folders named according to the model hyperparameters e.g. `32_0.2_25_1.0_0.0001_0.01_50` which contain the trained models with those hyperparameters. There is also a `.csv` file named `training_results.csv` tabulating the F1 score and accuracy for models trained with each set of hyperparameters.

## Evaluating trained models

To evaluate a grid trained models, run:

```
python3 model_evaluation.py sample_data trained_models/2021-9-20-1037-gru-fl ['Class 0','Class 1']
```

The `trained_models/2021-9-20-1037-gru-fl` is the folder where the trained model is saved, and `['Class 0','Class 1']` are the class labels (for a binary classification case). The labels need to be passed as a list of class labels within square parentheses separated by commas with no spaces, with each label enclosed within inverted commas ('). For example, ['Class 0', 'Class 1'], and [Class 0,Class 1] is incorrect. 

This will evaluate the models, and tabulate the model hyperparameters with F1 score and accuracy into `experiment_results.csv` within the configuration folder. It will also create two confusion matrices (evaluated on the validation and test set). 
