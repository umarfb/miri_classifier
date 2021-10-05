'''
Train a grid of models that take in mixed data (time-series + contextual)
and classify GOTO light curves
'''

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
import sys

import click
# import coloredlogs



# Create a GRU RNN model to take in mixed inputs
def create_gru_model(rnn_input_shape, dnn_input_shape, output_shape, 
                       rnn_dim, final_dense_dim, dropout, regularization, mask_value):
    
    # Create RNN branch
    rnn_input = tf.keras.layers.Input(shape=(rnn_input_shape))
    x1 = tf.keras.layers.Masking(input_shape=rnn_input_shape,
                                  mask_value=mask_value)(rnn_input)
    x1 = tf.keras.layers.GRU(rnn_dim, input_shape=rnn_input_shape,
                              return_sequences=True)(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.GRU(rnn_dim)(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.models.Model(inputs=rnn_input, outputs=x1)
    
    # Create DNN branch
    dense_input = tf.keras.layers.Input(shape=(dnn_input_shape))
    
    # Combine the output of the two models
    combined = tf.concat([x1.output, dense_input], axis=1)
    x3 = tf.keras.layers.Dense(final_dense_dim, activation='sigmoid')(combined)
    x3 = tf.keras.layers.Dropout(dropout)(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Dense(final_dense_dim, activation='sigmoid',
                               kernel_regularizer=tf.keras.regularizers.l2(regularization))(x3)
    x3 = tf.keras.layers.Dense(output_shape, activation='softmax')(x3)

    model = tf.keras.models.Model(inputs=[x1.input, dense_input], outputs=x3)
    
    return model

# Create a LSTM RNN model to take in mixed inputs
def create_lstm_model(rnn_input_shape, dnn_input_shape, output_shape, 
                       rnn_dim, final_dense_dim, dropout, regularization, mask_value):
    
    # Create RNN branch
    rnn_input = tf.keras.layers.Input(shape=(rnn_input_shape))
    x1 = tf.keras.layers.Masking(input_shape=rnn_input_shape,
                                  mask_value=mask_value)(rnn_input)
    x1 = tf.keras.layers.LSTM(rnn_dim, input_shape=rnn_input_shape,
                              return_sequences=True)(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LSTM(rnn_dim)(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.models.Model(inputs=rnn_input, outputs=x1)
    
    # Create DNN branch
    dense_input = tf.keras.layers.Input(shape=(dnn_input_shape))
    
    # Combine the output of the two models
    combined = tf.concat([x1.output, dense_input], axis=1)
    x3 = tf.keras.layers.Dense(final_dense_dim, activation='sigmoid')(combined)
    x3 = tf.keras.layers.Dropout(dropout)(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Dense(final_dense_dim, activation='sigmoid',
                               kernel_regularizer=tf.keras.regularizers.l2(regularization))(x3)
    x3 = tf.keras.layers.Dense(output_shape, activation='softmax')(x3)

    model = tf.keras.models.Model(inputs=[x1.input, dense_input], outputs=x3)
    
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
                        y_train, X_val, y_val, epochs, class_weights, param_grid, metrics, save_file_suffix, verbose):

    print('Model configurations: ', len(param_grid))

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

    root = 'trained_models/' + timestamp + save_file_suffix
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
        subdir = '/{}'.format(fname)
        os.makedirs(root + subdir + '/model_weights') # save model weights

        checkpoint_path = root + subdir + '/model_weights/training/cp.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=0)

        if rnn_type == 'GRU':
            model = create_gru_model(rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape, 
                                    output_shape=output_shape, rnn_dim=rnn_dim, 
                                    final_dense_dim=final_dense_dim, dropout=dropout, 
                                    regularization=reg_l2, mask_value=mask_val)
        elif rnn_type == 'LSTM':
            model = create_lstm_model(rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape, 
                                    output_shape=output_shape, rnn_dim=rnn_dim, 
                                    final_dense_dim=final_dense_dim, dropout=dropout, regularization=reg_l2, mask_value=mask_val)


        # Initialize optimizer, use the Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile the model
        model.compile(loss=focal_loss.FocalCrossEntropy(gamma=gamma, alpha=class_weights),
        optimizer=optimizer, metrics=metrics)
        
        # Train the model, and save training history
        history = model.fit([X_train[0], X_train[1]], y_train,
            validation_data=([X_val[0], X_val[1]], y_val),
            epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[cp_callback])

        # Save plot
        plot = plot_history(history)
        plot.savefig(root + subdir +  '/training_history_plot.pdf', dpi=300, bbox_inches='tight', format='pdf')

        # Calculate macro f1 score
        f1_macro = f1_score(np.argmax(y_val, axis=1),
                            np.argmax(model.predict(X_val), axis=1),
                            average='macro')

        # Get performance of trained model on validation set
        performance = rnntools.evaluate_model(model, X_val, y_val, batch_size, verbose=verbose)
        score = {'loss':performance[0],
                'accuracy':performance[1],
                'f1':performance[2],
                'f1_macro':f1_macro}

        # Save trained model and weights
        model.save(root + subdir +  '/trained_model.h5')

        # Save model training history
        df = pd.DataFrame(history.history)
        df.to_csv(root + subdir +  '/training_history.csv', index=None)

        scores.append(score)

    # Get time at the end of training
    train_stop = dt.datetime.now()
    train_time = train_stop - train_start
    print('{} models trained in {} seconds'.format(len(param_grid), train_time.seconds))

    # Get experiment results
    results = list()
    for p, s in list(zip(list(param_grid), scores)):

        row = {**p, **s}
        results.append(row)

    results = pd.DataFrame(results)

    # Write results to .csv file
    results.to_csv(root + '/training_results.csv', index=0)

@click.command()
@click.argument("data_dir")
def main(data_dir):
    """data_dir: Specify the directory where the training and validation data are stored"""

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
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # We want to input the following data into the model
    # - Light curve data: Time-series data, measurements of an objects brightness over time
    # - Contextual data: Position, proximity to any galaxies

    # Load preprocessed data
    X_train_time = np.load('data/{}/train_time_features.npy'.format(data_dir))
    X_train_contextual = np.load('data/{}/train_contextual_features.npy'.format(data_dir))
    y_train = np.load('data/{}/train_labels.npy'.format(data_dir))
    X_train = (X_train_time, X_train_contextual)

    X_val_time = np.load('data/{}/validation_time_features.npy'.format(data_dir))
    X_val_contextual = np.load('data/{}/validation_contextual_features.npy'.format(data_dir))
    y_val = np.load('data/{}/validation_labels.npy'.format(data_dir))
    X_val = (X_val_time, X_val_contextual)

    # Print input and output dimensions
    rnn_input_shape = (X_train[0].shape[1], X_train[0].shape[2])
    print('RNN input shape ', rnn_input_shape)

    dnn_input_shape = (X_train[1].shape[1])
    print('Dense network input shape ',dnn_input_shape)

    output_shape = y_train.shape[1]
    print('Output shape ', output_shape)

    # Define hyperparameter grid
    batch_sizes = [32]
    learning_rates = [1e-4]
    rnn_dims = [25]
    dropouts = [0.5]
    reg_l2s = [0.01]
    final_dense_dims = [25]
    gammas = [1.0]
    mask_val = -99.

    # Define parameters that won't change
    verbose, epochs = 1, 5
    
    # Metrics to measure accuracy and f1 score
    metrics = [tf.keras.metrics.CategoricalAccuracy(), rnntools.f1] 
    
    # Make class weights to account for class imbalance
    y_labels = [np.argmax(y) for y in y_train]
    class_size = [y_labels.count(c) for c in np.unique(y_labels)]
    class_weights = np.array([sum(class_size)/c for c in class_size])

    # Scale weights
    class_weights = np.array(class_weights) / len(list(class_weights))
    print('Class weights: ', class_weights)

    # Convert to dictionary
    class_weights = dict(zip(np.unique(y_labels), np.array(class_weights)))

    param_grid_fl = ParameterGrid({'batch_size':batch_sizes,
                                'learning_rate':learning_rates,
                                'rnn_dim':rnn_dims,
                                'dropout':dropouts,
                                'regularization_l2':reg_l2s,
                                'final_dense_dim':final_dense_dims,
                                'gamma':gammas})

    param_grid_ce = ParameterGrid({'batch_size':batch_sizes,
                                'learning_rate':learning_rates,
                                'rnn_dim':rnn_dims,
                                'dropout':dropouts,
                                'regularization_l2':reg_l2s,
                                'final_dense_dim':final_dense_dims,
                                'gamma':[0.0]})


    # Train GRU models with focal loss
    train_model_grid(rnn_type='GRU', rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape, 
                    output_shape=output_shape, mask_val=mask_val, X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val, epochs=epochs, class_weights=class_weights, 
                    param_grid=param_grid_fl, metrics=metrics, save_file_suffix='-gru-fl', verbose=verbose)

    # Train LSTM models with focal loss
    train_model_grid(rnn_type='LSTM', rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape, 
                     output_shape=output_shape, mask_val=mask_val, X_train=X_train, y_train=y_train,
                     X_val=X_val, y_val=y_val, epochs=epochs, class_weights=class_weights, 
                     param_grid=param_grid_fl, metrics=metrics, save_file_suffix='-lstm-fl', verbose=verbose)

    # Train GRU models with cross entropy
    train_model_grid(rnn_type='GRU', rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape, 
                    output_shape=output_shape, mask_val=mask_val, X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val, epochs=epochs, class_weights=class_weights, 
                    param_grid=param_grid_ce, metrics=metrics, save_file_suffix='-gru-wce', verbose=verbose)

    # Train LSTM models with cross entropy
    train_model_grid(rnn_type='LSTM', rnn_input_shape=rnn_input_shape, dnn_input_shape=dnn_input_shape, 
                    output_shape=output_shape, mask_val=mask_val, X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val, epochs=epochs, class_weights=class_weights, 
                    param_grid=param_grid_ce, metrics=metrics, save_file_suffix='-lstm-wce', verbose=verbose)

if __name__=='__main__':
    main()