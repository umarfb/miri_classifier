'''
Script to take a trained model and evaluate it on the validation set
'''
from typing import List
import os
import pandas as pd 
import tensorflow as tf
import numpy as np
from rnntools import rnntools
import focal_loss
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import click
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_evaluation")

# Method to get model parameters
def get_params(name):
    param_vals = name.split('_')
    param_names = ['batch_size',
                   'dnn_dense_dim',
                   'dropout',
                   'final_dense_dim',
                   'gamma',
                   'learning_rate',
                   'lstm_dim',
                   'regularization_l2']
    
    params = {param_names[i]:param_vals[i] for i in range(len(param_vals))}
    
    return params

def load_model(model_path, weights_path):
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Define metrics
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(), rnntools.f1]

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=focal_loss.FocalCrossEntropy(),
            metrics=metrics)
    
    # Load model weights
    # Expect_partial would silence the warnings.         
    # There are warnings because we are restoring a model that has training information but we are using it only for prediction and not training.
    model.load_weights(weights_path).expect_partial() 
    return model


def evaluate_model(model, X_val, y_val):
    # Evaluate model on validation set
    model_eval = rnntools.evaluate_model(model, X_val, y_val, batch_size=128, verbose=1)
    
    # Calculate macro f1 score
    f1_macro = f1_score(np.argmax(y_val, axis=1),
                        np.argmax(model.predict(X_val), axis=1),
                        average='macro')
    
    # Calulcate area under ROC curve
    if y_val.shape[1] == 2:
        auc_macro = roc_auc_score(np.argmax(y_val, axis=1),
                            model.predict(X_val)[:,0])

    else: 
        auc_macro = roc_auc_score(np.argmax(y_val, axis=1),
                            model.predict(X_val),
                            average='macro', multi_class='ovo')
    return model_eval, f1_macro, auc_macro

@click.command()
@click.option("-d", "--data-dir", required=True, help="Specify the directory where the data is stored")
@click.option("-m", "--model_dir", required=True, help="Specify the directory where the model is stored")
@click.option("-l", "--labels", required=True, multiple=True, help="Specify class labels - repeat this option for each class")
def main(data_dir:str, model_dir:str, labels:List[str]):
    logger.info("Loading data ...")

    X_val_time = np.load("data/{}/validation_time_features.npy".format(data_dir))
    X_val_contextual = np.load("data/{}/validation_contextual_features.npy".format(data_dir))
    y_val = np.load("data/{}/validation_labels.npy".format(data_dir))
    X_val = (X_val_time, X_val_contextual)

    X_test_time = np.load("data/{}/test_time_features.npy".format(data_dir))
    X_test_contextual = np.load("data/{}/test_contextual_features.npy".format(data_dir))
    y_test = np.load("data/{}/test_labels.npy".format(data_dir))
    X_test = (X_test_time, X_test_contextual)

    # Get directory where trained models are saved
    model_dir += '/'
    model_files = next(os.walk(model_dir))[1]

    logger.info(f"class labels:{labels}")

    # Define metrics
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(), rnntools.f1]

    # List to store evaluation results
    results = list()

    logger.info(f"y_val shape: {y_val.shape}")

    for model_file in model_files:
        # Get model parameters
        params = get_params(model_file)

        # Define path to models
        model_path = os.path.join(model_dir, model_file, "trained_model.h5")
        weights_path = os.path.join(model_dir, model_file, "model_weights", "training", "cp.ckpt")
        
        model = load_model(model_path, weights_path)
        model_eval, f1_macro, auc_macro = evaluate_model(model, X_val, y_val)
            
        # Get evaluation scores
        scores = {'loss':model_eval[0],
                'categorical_accuracy':model_eval[1],
                'f1':model_eval[2],
                'f1_macro':f1_macro,
                'auc_macro':auc_macro,
    }
        
        d = {**params, **scores}
        results.append(d)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(model_dir, "experiment_results.csv"), index=None)

    # Plot confusion matrix
    best_model = results.sort_values(by='auc_macro', ascending=False).iloc[0]
    best_model_name = [str(best_model[i]) for i in range(7)]

    model_path = os.path.join(model_dir, '_'.join(best_model_name), "trained_model.h5")
    weights_path = os.path.join(model_dir, '_'.join(best_model_name), "model_weights/training/cp.ckpt")

    m_best = tf.keras.models.load_model(model_path, compile=False)

    # Compile model
    m_best.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=focal_loss.FocalCrossEntropy(),
            metrics=metrics)

    # Load model weights
    # Expect_partial would silence the warnings.  
    # A better solution would be to only save the variables required for inference when training: saver = tf.train.Saver(tf.model_variables())  
    m_best.load_weights(weights_path).expect_partial() 

    # Predict validation set
    y_pred_val = m_best.predict(X_val)
    cm = rnntools.confusion_matrix(y_pred_val, y_val, labels)
    cm.savefig(os.path.join(model_dir, "{}class-confusion-matrix-val.pdf").format(y_val.shape[1]), format='pdf', dpi=300, bbox_inches='tight')

    # Predict test set
    y_pred_test = m_best.predict(X_test)
    cm = rnntools.confusion_matrix(y_pred_test, y_test, labels)
    cm.savefig(os.path.join(model_dir, "{}class-confusion-matrix-test.pdf").format(y_val.shape[1]), format='pdf', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()