'''

Module with methods to preprocess and format GOTO data for use with RNNs
for classification

'''

import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cf_mtx
from sklearn.metrics import f1_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras import backend as K

'''
Method to create an input vector and an output label for an individual
lightcurve

Arguments:
    srcs_row - Pandas dataframe row containing object data from sources table
    lc - Dataframe containing photometric data
    scale_time - boolean, if True, set first value of time to 0

Returns:
    input_vector - Matrix of input data for the RNN
    label - Object label
'''
def create_io_vectors(srcs_row, lc, scale_time=True):

    # Get R.A and Dec
    ra = srcs_row['ra']
    dec = srcs_row['dec']

    # Get host galaxy flags
    host_flag_arcmin = srcs_row['gal_match_1arcmin']
    host_flag_half_arcmin = srcs_row['gal_match_0.5arcmin']
    host_flag_arcsec = srcs_row['gal_match_2.5arcsec']

    # Convert host galaxy flag to binary indicator (0,1)
    if host_flag_arcmin == True:
        host_flag_arcmin_true = 1
        host_flag_arcmin_false = 0
    else:
        host_flag_arcmin_true = 0
        host_flag_arcmin_false = 1
    
    if host_flag_half_arcmin == True:
        host_flag_half_arcmin_true = 1
        host_flag_half_arcmin_false = 0
    else:
        host_flag_half_arcmin_true = 0
        host_flag_half_arcmin_false = 1
    
    if host_flag_arcsec == True:
        host_flag_arcsec_true = 1
        host_flag_arcsec_false = 0
    else:
        host_flag_arcsec_true = 0
        host_flag_arcsec_false = 1

    # Get photometric data (lightcurve)
    mag = lc['mag'].values
    mag_err = lc['mag_err'].values
    flux = lc['flux'].values
    flux_err = lc['flux_err'].values
    #bg_mean = lc['background_mean'].values
    #bg_rms = lc['background_rms'].values
    fwhm = lc['fwhm'].values
    timestep = lc['jd'].values

    # If scale_time = True, timestep becomes days since first observation
    if scale_time == True:

        if len(timestep) == 0:
            timestep = 0.0
        else:
            try:
                timestep = timestep - min(timestep)
            except ValueError:
                timestep = 0.0
    
    # Create input vector for RNN
    input_vector = list()

    for i in range(len(lc)):

        element = [timestep[i], mag[i], mag_err[i], flux[i], flux_err[i], fwhm[i],
         ra, dec, host_flag_arcmin_true, host_flag_arcmin_false,
         host_flag_half_arcmin_true, host_flag_half_arcmin_false, host_flag_arcsec_true,
         host_flag_arcsec_false,]
        input_vector.append(element)

    input_vector = np.array(input_vector)

    # Get object label
    label = srcs_row['label']

    return input_vector, label

'''
Method to convert categorical labels to numerical labels

Arguments:
    data - Vector of categorical labels for a sample of data
    labels - Unique class labels 

Returns:
    num_labels - Vector of numerical labels
'''
def cat_to_num(data, labels):

    num_labels = list()

    for item in data:
        if item in labels:
            num_labels.append(list(labels).index(item))
    
    return np.array(num_labels, dtype='float32')

'''
Method to one-hot encode numerical labels

Arguments:
    data - Vector of numerical labels
    labels - Unique class labels 

Returns:
    encoded_labels - One-hot encoded numerical labels
'''
def one_hot_encode(data, labels):

    num_labels = cat_to_num(data, labels)
    encoded_labels = tf.keras.utils.to_categorical(num_labels)

    return encoded_labels

'''
Method to format data for input into an RNN. Create input matrices and output
labels.

Arguments:
    srcs_df - Dataframe containing sources and their data
    phot_df - Dataframe containing photometric data

Returns:
    X - Input matrix for the RNN
    y - Vector of labels, one-hot encoded
'''
def format_data(srcs_df, phot_df, labels, verbose=False):

    X = list()
    y = list()

    for i, row in srcs_df.iterrows():

        # Get object ID
        obj_id = row['id']

        # Get object lightcurve
        lc = phot_df[phot_df['source_id']==obj_id].sort_values(by=['jd'],\
            ascending=True)
        lc = lc[lc['filter']=='L']
            
        input_mtx, label = create_io_vectors(row, lc, scale_time=True)
        X.append(input_mtx)
        y.append(label)

        if verbose == True:
            print('Formatting data {0}/{1}'.format(i+1, len(srcs_df)),\
                end='\r')

    if verbose == True:       
        print()

    # Convert categorical label to numerical
    #y = cat_to_num(y, labels)

    # One-hot encode labels
    y = one_hot_encode(y, labels)

    return X, y

'''
Method to oversample data

Arguments:
    X - Input data
    new_samples - Number of new samples to generate
    label - Class label

Returns:
    X_oversampled - A new oversampled dataset
    y - Labels
'''
def oversample(X, new_samples, label):
    
    # Randomly draw from X for a specified number of times
    X_new = list(np.random.choice(X, new_samples))
    
    # Create labels for oversampled data
    y = np.array([label] * len(X_new))
    
    return X_new, y

'''
Method to build a RNN with the following architecture

- Masking layer to recognize padded inputs
- LSTM layer
- LSTM layer
- Dropout layer for regularization
- Dense layer with sigmoid activation for classification
- Dense layer to predict class probabilities with a softmax activation 

Arguments:
    lstm_dim - int, Length of LSTM vector
    input_shape - tuple in the format (n_timesteps, n_features)
    dropout - float, Dropout fraction
    dense_dim - int, Number of neurons in dense layer
    output_dim - int, Number of neurons in final output layer

Returns:
    model - An untrained RNN model
'''
def build_lstm(lstm_dim, input_shape, dropout, dense_dim, output_dim, mask_value):

    model = tf.keras.Sequential([

        tf.keras.layers.Masking(input_shape=input_shape, mask_value=mask_value),

        tf.keras.layers.LSTM(lstm_dim, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LSTM(lstm_dim),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(dense_dim, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    return model

'''
Method to train a model, and return the training history

Arguments:
    X_train - Training data matrix
    y_train - Training data labels
    X_val - Validation data matrix
    y_val - Validation data labels
    model - Model to train
    loss - Loss function to optimize, use categorical cross entropy for
        classification
    learning_rate - Learning rate, controls the size of steps taken during
        gradient descent
    epochs - Maximum number of epochs to train for
    batch_size - Number of examples to pass through the model before
        updating weights
    verbose - Display training progress
    metrics - Metrics to measure during training
    class_weights - Weighting for imbalanced classes, alters contribution to
        loss function
    early_stopping - If set to True, set model to stop training if a criteria
        is met
    early_stopping_param - Criteria to decide early stopping

Retruns:
    model - Trained model
    history - Training history
'''
def train_model(X_train, y_train, X_val, y_val, model, loss, learning_rate,
    epochs, batch_size, verbose, metrics, class_weights=None, early_stopping=False,
    early_stopping_param=None, cp_callback=None, decay_learning_rate=False, decay_steps=None, decay_rate=None):

    if decay_learning_rate == False:
        # Initialize optimizer, use the Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        # Initialize an optimizer with a decaying learning rate
        initial_learning_rate = learning_rate
        lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate,
         staircase=False, name=None)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    if early_stopping == True:

        # Early stopping restores best weights (from previous epoch) if there
        # is not imporvement after a set number of epochs
        es = tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_param[0],
            patience=early_stopping_param[1],
            #verbose=verbose,
            restore_best_weights=True)
        
        # Train model with early stopping
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            callbacks=[cp_callback], class_weight=class_weights)
    
    else:

        # Train model without early stopping
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            class_weight=class_weights, callbacks=[cp_callback])

    return model, history

'''
Method to evaluate a trained model on a test set

Arguments:
    model - Trained model
    X_test - Test data input matrix
    y_test - Test data labels
    batch_size = Batch size

Returns:
    results - Results in the format (loss, metrics:[accuracy, f1])
'''
def evaluate_model(model, X_test, y_test, batch_size, verbose):

    results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

    return results

'''
Define the F1 score as a metric

Arguments:
    y_true - True labels
    y_pred - Labels predicted by the model

Returns:
    f1 score
'''
def f1(y_true, y_pred):
    
    #y_true = tf.math.argmax(y_true, axis=0)
    #y_true = tf.cast(y_true, tf.float32)    
    
    #y_pred = tf.math.argmax(y_pred, axis=0)
    #y_pred = tf.cast(y_pred, tf.float32)

    def name():
        
        return('f1')
    
    def recall(y_true, y_pred):
        '''
        Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        
        return recall

    def precision(y_true, y_pred):
        '''
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_weighted(y_true, y_pred):

    y_true = tf.make_ndarray(y_true)
    y_pred = tf.make_ndarray(y_pred)

    f1 = f1_score(y_true, y_pred, average='weighted')

    return f1

'''
Method to plot model loss over time during training
'''
def plot_loss(history, size=(7, 4.5)):

    plt.figure(figsize=size)
    plt.plot(history.history['loss'], color='g')
    plt.plot(history.history['val_loss'], color='r')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right', frameon=False)
    
    return plt

'''
Method to plot confusion matrix

Arguments:
    y_pred - Labels predicted by the model
    y_true - True labels
    labels - Unique class labels
    normalize - boolean, Option to normalize the matrix
    size - tuple, Size of plot, default is (8,8)
'''
def confusion_matrix(y_pred, y_true, labels, normalize=True, size=(8,8), title=None):

    cm_abs = cf_mtx(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    if normalize == True:
        cm = cm_abs.astype('float') / cm_abs.sum(axis=1, keepdims=True)
    
    # Define colourmap for plot
    cmap = plt.cm.Oranges

    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Create an axis to the right of ax. Width of cax will be 5% of ax and the
    # padding between cax and ax will be fixed at 0.05 in
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    ax.figure.colorbar(im, cax=cax)
    ax.set_aspect('equal')
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=labels, yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label')

    # Set the tick labels font
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    ax.set_title(title)
    ax.tick_params(axis='y', pad=10)
    ax.set_ylim(len(cm)-0.5, -0.5)    # Needed to fix a bug that cuts the plot

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right", rotation=0,
            rotation_mode="anchor")
    im.set_clim(0,1)
    #plt.setp(ax.get_yticklabels(), ha="center",
            #rotation=90, rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' #if normalize else 'd'
    abs_fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=16)
            ax.text(j, i+0.15, '({})'.format(cm_abs[i, j], abs_fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
            
    plt.tight_layout()

    return plt