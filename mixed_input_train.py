'''
Train a grid of models that take in mixed data (time-series + contextual)
and classify GOTO light curves
'''

from genericpath import exists
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from rnntools import rnntools
import focal_loss
import datetime as dt
import click
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mixted_input_train")

def create_recurrent_model(rnn_input_shape, dnn_input_shape, output_shape,
                           rnn_dim, final_dense_dim, dropout, regularization, mask_value,
                           use_gru_instead_of_lstm: bool = False, use_mixed_input: bool = True):

    # Create RNN branch
    rnn_input = tf.keras.layers.Input(shape=(rnn_input_shape))
    x1 = tf.keras.layers.Masking(input_shape=rnn_input_shape,
                                 mask_value=mask_value)(rnn_input)
    if use_gru_instead_of_lstm:
        x1 = tf.keras.layers.GRU(rnn_dim, input_shape=rnn_input_shape,
                                 return_sequences=True)(x1)
    else:
        x1 = tf.keras.layers.LSTM(rnn_dim, input_shape=rnn_input_shape,
                                  return_sequences=True)(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    if use_gru_instead_of_lstm:
        x1 = tf.keras.layers.GRU(rnn_dim)(x1)
    else:
        x1 = tf.keras.layers.LSTM(rnn_dim)(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.models.Model(inputs=rnn_input, outputs=x1)

    # If use_mixed_input = True, create a branch to take in additional metadata
    if use_mixed_input:
        # Create DNN branch
        dense_input = tf.keras.layers.Input(shape=(dnn_input_shape))

        # Combine the output of the two models
        combined = tf.concat([x1.output, dense_input], axis=1)
        x3 = tf.keras.layers.Dense(final_dense_dim, activation='sigmoid')(combined)
    else:
        x3 = tf.keras.layers.Dense(final_dense_dim, activation='sigmoid')(x1.output)

    x3 = tf.keras.layers.Dropout(dropout)(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Dense(final_dense_dim, activation='sigmoid',
                            kernel_regularizer=tf.keras.regularizers.l2(regularization))(x3)
    x3 = tf.keras.layers.Dense(output_shape, activation='softmax')(x3)

    if use_mixed_input:
        model = tf.keras.models.Model(inputs=[x1.input, dense_input], outputs=x3)
    else:
        model = tf.keras.models.Model(inputs=x1.input, outputs=x3)

    return model


# Plot training history
def plot_history(history):
    # Plot training & validation loss values
    plt.figure(figsize=(7, 4.5))
    plt.plot(history.history['loss'], color='b')
    plt.plot(history.history['val_loss'], color='r')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right', frameon=False)

    return plt


# Train a grid of models
def train_model_grid(rnn_type, rnn_input_shape, dnn_input_shape, output_shape, mask_val, X_train,
                     y_train, X_val, y_val, epochs, class_weights, param_grid, metrics, save_file_suffix, verbose,
                     use_mixed):
    logger.info(f"Model configurations: {len(param_grid)}")

    # Go through parameter grid, train model, and get performance results
    scores = list()

    # Create a directory to save working results
    time = dt.datetime.now()
    timestamp = '{}-{}-{}-{}{}'.format(
        time.year,
        time.month,
        time.day,
        time.hour,
        time.minute
    )

    # Create the folder if it does not exist
    root = os.path.join('trained_models', timestamp, save_file_suffix)
    if not os.path.exists(root):
        os.makedirs(root)

    # Get start time of training
    train_start = dt.datetime.now()

    for params in list(param_grid):
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        rnn_dim = params['rnn_dim']
        dropout = params['dropout']
        reg_l2 = params['regularization_l2']
        final_dense_dim = params['final_dense_dim']
        gamma = params['gamma']

        # Model params for file name
        ps = ['{}'.format(i) for i in list(params.values())]
        fname = '_'.join(ps)

        # Create directory to save model results
        os.makedirs(os.path.join(root, fname, 'model_weights'))  # save model weights

        checkpoint_path = os.path.join(root, fname, 'model_weights/training/cp.ckpt')


        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0)

        if rnn_type.upper() == 'GRU':
            use_gru = True
        elif rnn_type.upper() == 'LSTM':
            use_gru = False
        else:
            raise Exception(f"Invalid rnn_type, valid values: GRU and LSTM, got: {rnn_type}")

        if use_mixed == False:
            dnn_input_shape = None
            use_mixed = False

        model = create_recurrent_model(
            rnn_input_shape=rnn_input_shape,
            dnn_input_shape=dnn_input_shape,
            output_shape=output_shape,
            rnn_dim=rnn_dim,
            final_dense_dim=final_dense_dim,
            dropout=dropout,
            regularization=reg_l2,
            mask_value=mask_val,
            use_gru_instead_of_lstm=use_gru,
            use_mixed_input=use_mixed)

        # Initialize optimizer, use the Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile the model
        model.compile(loss=focal_loss.FocalCrossEntropy(gamma=gamma, alpha=class_weights),
                      optimizer=optimizer, metrics=metrics)

        # Train the model, and save training history
        if use_mixed == True:
            train_data = [X_train[0], X_train[1]]
            val_data = [X_val[0], X_val[1]]
        else:
            train_data = X_train
            val_data = X_val

        history = model.fit(train_data, y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[cp_callback])

        # Save plot
        plot = plot_history(history)
        plot.savefig(os.path.join(root, fname, 'training_history_plot.pdf'), dpi=300, bbox_inches='tight', format='pdf')

        # Calculate macro f1 score
        f1_macro = f1_score(np.argmax(y_val, axis=1),
                            np.argmax(model.predict(X_val), axis=1),
                            average='macro')

        # Get performance of trained model on validation set
        performance = rnntools.evaluate_model(model, X_val, y_val, batch_size, verbose=verbose)
        score = {'loss': performance[0],
                 'accuracy': performance[1],
                 'f1': performance[2],
                 'f1_macro': f1_macro}

        # Save trained model and weights
        model.save(os.path.join(root, fname, 'trained_model.h5'))

        # Save model training history
        df = pd.DataFrame(history.history)
        df.to_csv(os.path.join(root, fname, 'training_history.csv'), index=None)

        scores.append(score)

    # Get time at the end of training
    train_stop = dt.datetime.now()
    train_time = train_stop - train_start
    logger.info('{} models trained in {} seconds'.format(len(param_grid), train_time.seconds))

    # Get experiment results
    results = list()
    for p, s in list(zip(list(param_grid), scores)):
        row = {**p, **s}
        results.append(row)

    results = pd.DataFrame(results)

    # Write results to .csv file
    results.to_csv(os.path.join(root, 'training_results.csv'), index=0)

@click.command()
@click.option("-d", "--data-dir", required=True, help="Specify the directory where the training and validation data are stored")
@click.option("-c", "--config-yaml", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Config YAML file for grid search")
@click.option("-v", "--verbose", count=True, default=1)
@click.option("-e", "--epochs", type=int, default=5)
@click.option("-mi", "--mixed", required=True, type=bool, help="Specify whether to use the mixed input version or not")
def main(data_dir: str, config_yaml: str, verbose: int, epochs: int, mixed: bool):
    # Enable XLA
    tf.config.optimizer.set_jit(True)

    # Limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.exception("Memory issue")

    # We want to input the following data into the model
    # - Light curve data: Time-series data, measurements of an objects brightness over time
    # - Contextual data: Position, proximity to any galaxies

    # Load preprocessed data
    X_train_time = np.load('data/{}/train_time_features.npy'.format(data_dir))
    if mixed == True: # Load in additional metadata if mixed flag is true
        X_train_contextual = np.load('data/{}/train_contextual_features.npy'.format(data_dir))
        X_train = (X_train_time, X_train_contextual)
    else:
        X_train = X_train_time
    y_train = np.load('data/{}/train_labels.npy'.format(data_dir))

    X_val_time = np.load('data/{}/validation_time_features.npy'.format(data_dir))
    if mixed == True: # Load in additional metadata if mixed flag is true
        X_val_contextual = np.load('data/{}/validation_contextual_features.npy'.format(data_dir))
        X_val = (X_val_time, X_val_contextual)
    else:
        X_val = X_val_time
    y_val = np.load('data/{}/validation_labels.npy'.format(data_dir))

    if mixed == True:
        # Print input and output dimensions
        rnn_input_shape = (X_train[0].shape[1], X_train[0].shape[2])
        logger.info(f"RNN input shape {rnn_input_shape}")

        dnn_input_shape = (X_train[1].shape[1])
        logger.info(f"Dense network input shape {dnn_input_shape}")
    else:
        rnn_input_shape = (X_train.shape[1], X_train.shape[2])
        dnn_input_shape = None
        logger.info(f"RNN input shape {rnn_input_shape}")

    output_shape = y_train.shape[1]
    logger.info(f"Output shape {output_shape}")

    logger.info(f"Loading config file {config_yaml}...")
    config = yaml.load(open(config_yaml), Loader=yaml.FullLoader)
    # Load hyperparameter grid
    batch_sizes = config["batch_sizes"]
    learning_rates = config["learning_rates"]
    rnn_dims = config["rnn_dims"]
    dropouts = config["dropouts"]
    reg_l2s = config["reg_l2s"]
    final_dense_dims = config["final_dense_dims"]
    gammas = config["gammas"]
    mask_val = config["mask_val"]

    # Metrics to measure accuracy and f1 score
    metrics = [tf.keras.metrics.CategoricalAccuracy(), rnntools.f1]

    # Make class weights to account for class imbalance
    y_labels = [np.argmax(y) for y in y_train]
    class_size = [y_labels.count(c) for c in np.unique(y_labels)]
    class_weights = np.array([sum(class_size) / c for c in class_size])

    # Scale weights, divide by number of classes to make sure the values are not too large
    class_weights = np.array(class_weights) / len(list(class_weights))
    logger.info(f"Class weights: {class_weights}")

    # Convert to dictionary
    class_weights = dict(zip(np.unique(y_labels), np.array(class_weights)))

    param_grid_fl = ParameterGrid({'batch_size': batch_sizes,
                                   'learning_rate': learning_rates,
                                   'rnn_dim': rnn_dims,
                                   'dropout': dropouts,
                                   'regularization_l2': reg_l2s,
                                   'final_dense_dim': final_dense_dims,
                                   'gamma': gammas})

    # For the focal loss function, a gamma parameter of zero reduces the focal loss
    # to the cross entropy loss function
    param_grid_ce = ParameterGrid({'batch_size': batch_sizes,
                                   'learning_rate': learning_rates,
                                   'rnn_dim': rnn_dims,
                                   'dropout': dropouts,
                                   'regularization_l2': reg_l2s,
                                   'final_dense_dim': final_dense_dims,
                                   'gamma': [0.0]}) 

    # Train GRU models with focal loss
    train_model_grid(rnn_type='GRU', rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape,
                     output_shape=output_shape, mask_val=mask_val, X_train=X_train, y_train=y_train,
                     X_val=X_val, y_val=y_val, epochs=epochs, class_weights=class_weights,
                     param_grid=param_grid_fl, metrics=metrics, save_file_suffix='-gru-fl', verbose=verbose,
                     use_mixed=mixed)

    # Train LSTM models with focal loss
    train_model_grid(rnn_type='LSTM', rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape,
                     output_shape=output_shape, mask_val=mask_val, X_train=X_train, y_train=y_train,
                     X_val=X_val, y_val=y_val, epochs=epochs, class_weights=class_weights,
                     param_grid=param_grid_fl, metrics=metrics, save_file_suffix='-lstm-fl', verbose=verbose,
                     use_mixed=mixed)

    # Train GRU models with cross entropy
    train_model_grid(rnn_type='GRU', rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape,
                     output_shape=output_shape, mask_val=mask_val, X_train=X_train, y_train=y_train,
                     X_val=X_val, y_val=y_val, epochs=epochs, class_weights=class_weights,
                     param_grid=param_grid_ce, metrics=metrics, save_file_suffix='-gru-wce', verbose=verbose,
                     use_mixed=mixed)

    # Train LSTM models with cross entropy
    train_model_grid(rnn_type='LSTM', rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape,
                     output_shape=output_shape, mask_val=mask_val, X_train=X_train, y_train=y_train,
                     X_val=X_val, y_val=y_val, epochs=epochs, class_weights=class_weights,
                     param_grid=param_grid_ce, metrics=metrics, save_file_suffix='-lstm-wce', verbose=verbose,
                     use_mixed=mixed)


if __name__ == '__main__':
    main()
