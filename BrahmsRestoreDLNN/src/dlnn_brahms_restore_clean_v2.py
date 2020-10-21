# Cleaned version of as of 10/20/20

from scipy.io import wavfile
import scipy.signal as sg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Lambda, TimeDistributed, Layer, LSTM, Bidirectional, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import datetime
import numpy as np
import math
import random
import json
import os
import sys



# Loss function
def discriminative_loss(piano_true, noise_true, piano_pred, noise_pred, loss_const):
    last_dim = piano_pred.shape[1] * piano_pred.shape[2]
    return (
        tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
        (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
        tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) -
        (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1))
    )



def make_model(features, sequences, name='Model'):

    input_layer = Input(shape=(sequences, features), dtype='float32', 
                        name='piano_noise_mixed')
    piano_true = Input(shape=(sequences, features), dtype='float32', 
                       name='piano_true')
    noise_true = Input(shape=(sequences, features), dtype='float32', 
                       name='noise_true')

    x = SimpleRNN(features // 2, 
                  activation='relu', 
                  return_sequences=True) (input_layer) 
    piano_pred = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    noise_pred = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
  
    model = Model(inputs=[input_layer, piano_true, noise_true],
                  outputs=[piano_pred, noise_pred])

    return model



# Model "wrapper" for many-input loss function
class RestorationModel2(Model):
    def __init__(self, model, loss_const):
        super(RestorationModel2, self).__init__()
        self.model = model
        self.loss_const = loss_const
       
    def call(self, inputs):
        return self.model(inputs)

    def compile(self, optimizer, loss):
        super(RestorationModel2, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        # Unpack data - what generator yeilds
        x, piano_true, noise_true = data

        with tf.GradientTape() as tape:
            piano_pred, noise_pred = self.model((x, piano_true, noise_true), training=True)
            loss = self.loss(piano_true, noise_true, piano_pred, noise_pred, self.loss_const)

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {'loss': loss}

    def test_step(self, data):
        x, piano_true, noise_true = data

        piano_pred, noise_pred = self.model((x, piano_true, noise_true), training=False)
        loss = self.loss(piano_true, noise_true, piano_pred, noise_pred, self.loss_const)
        
        return {'loss': loss}



def make_imp_model(features, sequences, loss_const=0.05, 
                   optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.7),
                   name='Restoration Model', epsilon=10 ** (-10)):
    
    # NEW Semi-imperative model
    model = RestorationModel2(make_model(features, sequences, name='Training Model'),
                              loss_const=loss_const)

    model.compile(optimizer=optimizer, loss=discriminative_loss)

    return model



# MODEL TRAIN & EVAL FUNCTION
def evaluate_source_sep(train_generator, validation_generator,
                        num_train, num_val, n_feat, n_seq, batch_size, 
                        loss_const, epochs=20, 
                        optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.75),
                        patience=10, epsilon=10 ** (-10)):
   
    print('Making model...')    # IMPERATIVE MODEL - Customize Fit
    model = make_imp_model(n_feat, n_seq, loss_const=loss_const, optimizer=optimizer, epsilon=epsilon)
    
    print('Going into training now...')
    hist = model.fit(train_generator,
                     steps_per_epoch=math.ceil(num_train / batch_size),
                     epochs=epochs,
                     validation_data=validation_generator,
                     validation_steps=math.ceil(num_val / batch_size),
                     callbacks=[EarlyStopping('val_loss', patience=patience, mode='min')])
    print(model.summary())



# NEURAL NETWORK DATA GENERATOR
def my_dummy_generator(num_samples, batch_size, train_seq, train_feat):

    while True:
        for offset in range(0, num_samples, batch_size):

            # Initialise x, y1 and y2 arrays for this batch
            x, y1, y2 = (np.empty((batch_size, train_seq, train_feat)),
                            np.empty((batch_size, train_seq, train_feat)),
                            np.empty((batch_size, train_seq, train_feat)))

            yield (x, y1, y2)



def main():
    epsilon = 10 ** (-10)
    train_batch_size = 5
    loss_const, epochs, val_split = 0.05, 10, 0.25
    optimizer = tf.keras.optimizers.RMSprop(clipvalue=0.9)

    TRAIN_SEQ_LEN, TRAIN_FEAT_LEN = 1847, 2049
    TOTAL_SMPLS = 60 

    # Validation & Training Split
    indices = list(range(TOTAL_SMPLS))
    val_indices = indices[:math.ceil(TOTAL_SMPLS * val_split)]
    num_val = len(val_indices)
    num_train = TOTAL_SMPLS - num_val
   
    train_seq, train_feat = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN
    print('Train Input Stats:')
    print('N Feat:', train_feat, 'Seq Len:', train_seq, 'Batch Size:', train_batch_size)

    # Create data generators and evaluate model with them
    train_generator = my_dummy_generator(num_train,
                        batch_size=train_batch_size, train_seq=train_seq,
                        train_feat=train_feat)
    validation_generator = my_dummy_generator(num_val,
                        batch_size=train_batch_size, train_seq=train_seq,
                        train_feat=train_feat)

    evaluate_source_sep(train_generator, validation_generator, num_train, num_val,
                            n_feat=train_feat, n_seq=train_seq, 
                            batch_size=train_batch_size, 
                            loss_const=loss_const, epochs=epochs,
                            optimizer=optimizer, epsilon=epsilon)

if __name__ == '__main__':
    main()

