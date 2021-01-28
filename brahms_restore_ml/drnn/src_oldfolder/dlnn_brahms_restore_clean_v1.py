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



# def make_model(features, sequences, name='Model'):

#     input_layer = Input(shape=(sequences, features), dtype='float32', 
#                         name='piano_noise_mixed')
#     piano_true = Input(shape=(sequences, features), dtype='float32', 
#                        name='piano_true')
#     noise_true = Input(shape=(sequences, features), dtype='float32', 
#                        name='noise_true')

#     x = SimpleRNN(features // 2, 
#                   activation='relu', 
#                   return_sequences=True) (input_layer) 
#     piano_pred = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
#     noise_pred = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
  
#     model = Model(inputs=[input_layer, piano_true, noise_true],
#                   outputs=[piano_pred, noise_pred])

#     return model

# Imperative model
class RestorationModel(Model):
    def __init__(self, features, loss_const, name='Restoration Model', 
                 epsilon=10**(-10)):#, **kwargs):
        # super(RestorationModel, self).__init__(name=name, **kwargs)
        super(RestorationModel, self).__init__()
        # self.config = config
        self.loss_const = loss_const
        # self._name = name
        # if self.config is not None:
        #     pass
        # else:
        self.rnn1 = SimpleRNN(features // 2, 
                                activation='relu', 
                                return_sequences=True)
        self.rnn2 = SimpleRNN(features // 2, 
                                activation='relu', 
                                return_sequences=True)
        self.dense_branch1 = TimeDistributed(Dense(features), name='piano_hat')
        self.dense_branch2 = TimeDistributed(Dense(features), name='noise_hat')
        # self.piano_tf_mask = TimeFreqMasking(epsilon=epsilon, name='piano_pred')
        # self.noise_tf_mask = TimeFreqMasking(epsilon=epsilon, name='noise_pred')

    def call(self, inputs):
        # print('\nINPUTS:')
        # print(inputs)
        # print('\n')

        print('\nINPUT (index 0):')
        print(inputs[0])
        print('\n')
        # print('INPUTS CONTENTS:')
        # print(inputs[0])
        # print(inputs[1])
        # print(inputs[2])
        # batch_input1, batch_input2, batch_input3 = inputs[0][0], inputs[1][0], inputs[2][0]

        # model_input = tf.convert_to_tensor([inputs[0][0], inputs[1][0], inputs[2][0]], dtype=tf.float32)
        # model_input = tf.convert_to_tensor([inputs[0], inputs[1], inputs[2]], dtype=tf.float32)
        # model_input = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # model_input = tf.constant([inputs[0], inputs[1], inputs[2]])
        # print('\nMODEL INPUT:')
        # print(model_input)
        # print('\n')
        # piano_noise_mix, p_true, n_true = inputs[0], inputs[1], inputs[2]
        # print('SHAPE OF INCOMING INPUTS (CALL):', inputs.shape)
        # global_phases1, global_phases2, global_phases3 = np.fromfile('1').reshape((1847, 2049)), np.fromfile('2').reshape((1847, 2049)), np.fromfile('3').reshape((1847, 2049))
        # print('TEST - WRITE THESE TO WAV -', global_phases1.shape, global_phases2.shape, global_phases3.shape)
        # synthetic_sig = make_synthetic_signal(inputs[0].numpy(), global_phases1, PIANO_WDW_SIZE, 
        #                                       'int16', ova=True, debug=False)
        # wavfile.write('1.wav', 44100, synthetic_sig)
        # synthetic_sig = make_synthetic_signal(inputs[1].numpy(), global_phases2, PIANO_WDW_SIZE, 
        #                                       'int16', ova=True, debug=False)
        # wavfile.write('2.wav', 44100, synthetic_sig)
        # synthetic_sig = make_synthetic_signal(inputs[2].numpy(), global_phases3, PIANO_WDW_SIZE, 
        #                                       'int16', ova=True, debug=False)
        # wavfile.write('3.wav', 44100, synthetic_sig)
        # print('SHAPE OF X INPUT (CALL):', piano_noise_mix.shape)
        # if self.config is not None:
        #     pass
        # else:
        # x = self.rnn1(piano_noise_mix)
        x = self.rnn1(inputs[0])
        # x = self.rnn1(model_input)
        x = self.rnn2(x)
        piano_pred = self.dense_branch1(x)   # source 1 branch
        noise_pred = self.dense_branch2(x)   # source 2 branch
        # piano_hat = self.dense_branch1(x)   # source 1 branch
        # noise_hat = self.dense_branch2(x)   # source 2 branch
        # piano_pred = self.piano_tf_mask([piano_hat, noise_hat, inputs])
        # noise_pred = self.noise_tf_mask([noise_hat, piano_hat, inputs])
        # piano_pred = self.piano_tf_mask([piano_hat, noise_hat, piano_noise_mix])
        # noise_pred = self.noise_tf_mask([noise_hat, piano_hat, piano_noise_mix])

        return (piano_pred, noise_pred)

    def compile(self, optimizer, loss):
        super(RestorationModel, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        # Unpack data - what generator yeilds
        # {'piano_noise_mixed': x, 'piano_true': y1, 'noise_true': y2}, y1, y2 = data
        x, piano_true, noise_true = data

        # print('SHAPE OF X INPUT (TRAIN_STEP):', x.shape)
        with tf.GradientTape() as tape:
            # y_pred = self(x, training=True) # Forward pass
            piano_pred, noise_pred = self((x, piano_true, noise_true), training=True)   # Forward pass
            # Compute the loss value
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # loss = self.compiled_loss(piano_true, noise_true, piano_pred, noise_pred, self.loss_const)
            loss = self.loss(piano_true, noise_true, piano_pred, noise_pred, self.loss_const)


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Uncomment if error here (no metrics)
        # self.compiled_metrics.update_state(piano_true, noise_true, piano_pred, noise_pred)
        # Return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}
        return {'loss': loss}

    def test_step(self, data):
        x, piano_true, noise_true = data

        piano_pred, noise_pred = self((x, piano_true, noise_true), training=False)
        loss = self.loss(piano_true, noise_true, piano_pred, noise_pred, self.loss_const)
        
        return {'loss': loss}

# # Model "wrapper" for many-input loss function
# class RestorationModel2(Model):
#     def __init__(self, model, loss_const):
#         super(RestorationModel2, self).__init__()
#         self.model = model
#         self.loss_const = loss_const
       
#     def call(self, inputs):
#         return self.model(inputs)

#     def compile(self, optimizer, loss):
#         super(RestorationModel2, self).compile()
#         self.optimizer = optimizer
#         self.loss = loss

#     def train_step(self, data):
#         # Unpack data - what generator yeilds
#         x, piano_true, noise_true = data

#         with tf.GradientTape() as tape:
#             piano_pred, noise_pred = self.model((x, piano_true, noise_true), training=True)
#             loss = self.loss(piano_true, noise_true, piano_pred, noise_pred, self.loss_const)

#         trainable_vars = self.model.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
#         return {'loss': loss}

#     def test_step(self, data):
#         x, piano_true, noise_true = data

#         piano_pred, noise_pred = self.model((x, piano_true, noise_true), training=False)
#         loss = self.loss(piano_true, noise_true, piano_pred, noise_pred, self.loss_const)
        
#         return {'loss': loss}



# def make_imp_model(features, sequences, loss_const=0.05, 
#                    optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.7),
#                    name='Restoration Model', epsilon=10 ** (-10)):
def make_imp_model(features, loss_const=0.05, 
                   optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.7),
                   name='Restoration Model', epsilon=10 ** (-10)):
    
    # NEW Semi-imperative model
    # model = RestorationModel2(make_model(features, sequences, name='Training Model'),
    #                           loss_const=loss_const)
    # Imperative model
    model = RestorationModel(features, loss_const, name='Training Model', epsilon=epsilon)

    model.compile(optimizer=optimizer, loss=discriminative_loss)

    return model



# MODEL TRAIN & EVAL FUNCTION
def evaluate_source_sep(train_generator, validation_generator,
                        num_train, num_val, n_feat, n_seq, batch_size, 
                        loss_const, epochs=20, 
                        optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.75),
                        patience=10, epsilon=10 ** (-10)):
   
    print('Making model...')    # IMPERATIVE MODEL - Customize Fit
    # model = make_imp_model(n_feat, n_seq, loss_const=loss_const, optimizer=optimizer, epsilon=epsilon)
    model = make_imp_model(n_feat, loss_const=loss_const, optimizer=optimizer, epsilon=epsilon)

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

