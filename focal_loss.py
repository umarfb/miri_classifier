'''
Class for focal loss function with multiclass classification
'''
import numpy as np
import tensorflow as tf

# Create focal loss class
class FocalCrossEntropy(tf.keras.losses.Loss):
    
    '''
    Args:
        gamma: Modulating factor, default to 2.0
        alpha: Class weights, default is None
        
    Returns:
        Focal loss
    
    Raises:
        ValueError: gamma >= 0.
    '''
    def __init__(self, gamma=2.0, alpha=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='focal_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        
        if self.gamma and self.gamma < 0.:
            raise ValueError('Gamma must be equal to or greater than zero')
        
        # Calculate categorical cross-entropy
        ce = tf.losses.categorical_crossentropy(y_true, y_pred,
                                                from_logits=self.from_logits)
        
        # Calculate gamma factor
        gamma_factor = 1 - tf.reduce_max(y_true * y_pred, axis=1)
        gamma_factor = tf.pow(gamma_factor, self.gamma)
        
        if self.alpha != None:
            # Get class weights
            cw = tf.convert_to_tensor(np.array(list(self.alpha.values())),
                                     dtype=y_true.dtype)
            cw = tf.reduce_max(cw * y_true, axis=1)
            fl = cw * gamma_factor * ce
        else:
            fl = gamma_factor * ce
        
        return tf.reduce_mean(fl) 
    
    def get_config(self):
        
        config = {'gamma':self.gamma,
                  'from_logits':self.from_logits}
        
        return config