# Quinn Coleman - B.M.S. Thesis Brahms Restoration
# Advisor: Dr. Dennis Sun
# 8/31/20
# dlnn_brahms_restore - neural network to restore brahms recording
# Custom training loop version

# DATA RULES #
# - If writing a transformed signal, write it back using its original data type/range (wavfile lib)
# - Convert signals into float64 for processing (numpy default, no GPUs usit ed) (in make_spgm() do a check)
# - Convert data fed into NN into float32 (GPUs like it)
# - No functionality to train on 8-bit PCM signal (unsigned) b/c of rare case
##############

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import os
from copy import deepcopy
import multiprocessing
import time
from . import data
# from .data import *
from ..audio_data_processing import make_spectrogram, make_synthetic_signal, plot_matrix, SPGM_BRAHMS_RATIO

# NEURAL NETWORK FUNCTIONS

# MODEL TRAIN & EVAL FUNCTION - Training Loop From Scratch
def evaluate_source_sep(# train_dataset, val_dataset,
                        # train_generator, validation_generator,

            x_train_files, y1_train_files, y2_train_files,
            x_val_files, y1_val_files, y2_val_files,

                        # train_step_func, test_step_func,
                        num_train, num_val, n_feat, n_seq, batch_size, 
                        loss_const, epochs=20, 
                        opt_name='RMSProp', opt_clip_val=-1, opt_lr=0.001,
                        # optimizer=None,
                        # optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.75),
                        patience=100, epsilon=10 ** (-10), config=None, recent_model_path=None, pc_run=False,
                        t_mean=None, t_std=None, grid_search_iter=None, gs_path=None, combos=None, gs_id='',
                        ret_queue=None, dataset2=False, data_path=None, pad_len=-1):

    from .model import make_model
    import tensorflow as tf
    # # from tensorflow import keras
    # # from tensorflow.keras.losses import Loss
    # from tensorflow.keras.layers import Input, SimpleRNN, Dense, Lambda, TimeDistributed, Layer, LSTM, Bidirectional, BatchNormalization, Concatenate, Activation
    # from tensorflow.keras.models import Model
    # # from tensorflow.keras.utils import Sequence
    # # from tensorflow.keras.activations import relu
    # from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


    physical_gpus = tf.config.list_physical_devices('GPU')
    if (not pc_run) and physical_gpus:
        # # No courtesy, allocate all GPU mem for me only, guarantees no else can mess up me. If no avail GPUs, admin's fault
        # # mostly courtesy to others on F35 system
        # print("Setting memory growth on GPUs")
        # for i in range(len(physical_gpus)):
        #     tf.config.experimental.set_memory_growth(physical_gpus[i], True)

        # Restrict TensorFlow to only use one GPU (exclusive access w/ mem growth = False), F35 courtesy
        try:
            # Pick random GPU, 1/10 times, pick GPU 0 other times 
            # (pick same GPU most times, but be prepared for it to be taken between restarts)
            # choice = random.randint(0, 9)
            # chosen_gpu = random.randint(0, len(physical_gpus)-1) if (choice == 0) else 0
            chosen_gpu = 0
            print("Restricting TF run to only use 1 GPU:", physical_gpus[chosen_gpu], "\n")
            tf.config.experimental.set_visible_devices(physical_gpus[chosen_gpu], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # BLOCK MUST BE MADE AFTER SET MEM GROWTH ABOVE
    print('Tensorflow version:', tf.__version__)
    # Tells tf.function not to make graph, & run all ops eagerly (step debugging)
    # tf.config.run_functions_eagerly(True)             # For nightly release
    # tf.config.experimental_run_functions_eagerly(True)  # For TF 2.2 (non-nightly)
    print('Eager execution enabled? (default)', tf.executing_eagerly())

    # Instantiate optimizer
    optimizer = (tf.keras.optimizers.RMSprop(clipvalue=opt_clip_val, learning_rate=opt_lr) if 
                    opt_name == 'RMSprop' else
                tf.keras.optimizers.Adam(clipvalue=opt_clip_val, learning_rate=opt_lr))

    # class Standardize(Layer):
    #     def __init__(self, mean, std, **kwargs):
    #         super(Standardize, self).__init__(**kwargs)
    #         self.mean = mean
    #         self.std = std

    #     def call(self, input):
    #         input -= self.mean
    #         input /= self.std
    #         return input


    # class UnStandardize(Layer):
    #     def __init__(self, mean, std, **kwargs):
    #         super(UnStandardize, self).__init__(**kwargs)
    #         self.mean = mean
    #         self.std = std

    #     def call(self, input):
    #         input *= self.std
    #         input += self.mean
    #         return input


    # # TF Masking layer has too compilcated operations for a lambda, and want to serialize model
    # class TimeFreqMasking(Layer):

    #     # Init is for input-independent variables
    #     # def __init__(self, piano_flag, **kwargs):
    #     def __init__(self, epsilon, **kwargs):
    #         # MAKE LAYER DEAL IN FLOAT16
    #         # TEST FLOAT16
    #         # kwargs['autocast'] = False
    #         # MIXED PRECISION - output layer needs to produce float32
    #         # kwargs['dtype'] = 'float32' # - or actually try in __init__ below
    #         # super(TimeFreqMasking, self).__init__(dtype='float32', **kwargs)
    #         super(TimeFreqMasking, self).__init__(**kwargs)
    #         # self.piano_flag = piano_flag
    #         self.epsilon = epsilon

    #     # No build method, b/c passing in multiple inputs to layer (no single shape)

    #     def call(self, inputs):
    #         # Try this alternative format if below doesn't work
    #         # self.total = tf.Variable(initial_value=y_hat_other, trainable=False)
    #         # self.total.assign_add(tf.reduce_sum(inputs, axis=0))
    #         # return self.total

    #         # y_hat_self, y_hat_other, x_mixed = inputs[0], inputs[1], inputs[2]
    #         y_hat_self, y_hat_other, x_mixed = inputs

    #         # print('TYPES IN TF MASKING:', y_hat_self.dtype, y_hat_other.dtype, x_mixed.dtype)

    #         mask = tf.abs(y_hat_self) / (tf.abs(y_hat_self) + tf.abs(y_hat_other) + self.epsilon)
    #         # print('Mask Shape:', mask.shape)
    #         # ones = tf.convert_to_tensor(np.ones(mask.shape).astype('float32'))
    #         # print('Ones Shape:', ones.shape)
    #         # y_tilde_self = mask * x_mixed if (self.piano_flag) else (ones - mask) * x_mixed
    #         y_tilde_self = mask * x_mixed

    #         # print('Y Tilde Shape:', y_tilde_self.shape)
    #         return y_tilde_self
        
    #     # config only contains things in __init__
    #     def get_config(self):
    #         config = super(TimeFreqMasking, self).get_config()
    #         config.update({'epsilon': self.epsilon})
    #         return config
        
    #     def from_config(cls, config):
    #         return cls(**config)


    # def discrim_loss(y_true, y_pred):
    #     piano_true, noise_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    #     loss_const = y_pred[-1, :, :][0][0]
    #     piano_pred, noise_pred = tf.split(y_pred[:-1, :, :], num_or_size_splits=2, axis=0)

    #     # True sum of L2 Norm doesn't divide by N (want that)
    #     last_dim = piano_pred.shape[1] * piano_pred.shape[2]
    #     return (
    #         tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
    #         (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
    #         tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) -
    #         (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1))
    #     )

    # def make_model(features, sequences, name='Model', epsilon=10 ** (-10),
    #                     loss_const=0.05, config=None, t_mean=None, t_std=None, 
    #                     optimizer=tf.keras.optimizers.RMSprop(),
    #                     pre_trained_wgts=None,
    #                     # GPU mem as func of HP TEST
    #                     # test=16, 
    #                     test=0, 
    #                     pc_run=False,
    #                     keras_fit=False):
    #     # TEST FLOAT16
    #     # MIXED PRECISION
    #     input_layer = Input(shape=(sequences, features), name='piano_noise_mixed')
    #     # input_layer = Input(shape=(sequences, features), dtype='float16', 
    #     #                     name='piano_noise_mixed')

    #     if config is not None:
    #         num_layers = len(config['layers'])
    #         prev_layer_type = None  # Works b/c all RNN stacks are size > 1
    #         for i in range(num_layers):
    #             layer_config = config['layers'][i]
    #             curr_layer_type = layer_config['type']

    #             # Standardize option
    #             if config['scale'] and i == 0:
    #                 x = Standardize(t_mean, t_std) (input_layer)

    #             # Add skip connection if necessary
    #             if (config['rnn_res_cntn'] and prev_layer_type is not None and
    #                 prev_layer_type != 'Dense' and curr_layer_type == 'Dense'):
    #                 x = Concatenate() ([x, input_layer])
        
    #             if curr_layer_type == 'RNN':
    #                 if config['bidir']:
    #                     x = Bidirectional(SimpleRNN(features // layer_config['nrn_div'], 
    #                             activation=layer_config['act'], 
    #                             use_bias=config['bias_rnn'],
    #                             dropout=config['rnn_dropout'][0],
    #                             recurrent_dropout=config['rnn_dropout'][1],
    #                             return_sequences=True)) (input_layer if (i == 0 and not config['scale']) else x)
    #                 else:
    #                     x = SimpleRNN(features // layer_config['nrn_div'], 
    #                             activation=layer_config['act'], 
    #                             use_bias=config['bias_rnn'],
    #                             dropout=config['rnn_dropout'][0],
    #                             recurrent_dropout=config['rnn_dropout'][1],
    #                             return_sequences=True) (input_layer if (i == 0 and not config['scale']) else x)

    #             elif curr_layer_type == 'LSTM':
    #                 if config['bidir']:
    #                     x = Bidirectional(LSTM(features // layer_config['nrn_div'], 
    #                             activation=layer_config['act'], 
    #                             use_bias=config['bias_rnn'],
    #                             dropout=config['rnn_dropout'][0],
    #                             recurrent_dropout=config['rnn_dropout'][1],
    #                             return_sequences=True)) (input_layer if (i == 0 and not config['scale']) else x)
    #                 else:
    #                     x = LSTM(features // layer_config['nrn_div'], 
    #                             activation=layer_config['act'], 
    #                             use_bias=config['bias_rnn'],
    #                             dropout=config['rnn_dropout'][0],
    #                             recurrent_dropout=config['rnn_dropout'][1],
    #                             return_sequences=True) (input_layer if (i == 0 and not config['scale']) else x)
    #             elif curr_layer_type == 'Dense':
    #                 if i == (num_layers - 1):   # Last layer is fork layer
    #                     # Reverse standardization at end of model if appropriate
    #                     if config['scale']:
    #                         piano_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
    #                                                         activation=layer_config['act'], 
    #                                                         use_bias=config['bias_dense']), 
    #                                                     name='piano_hat'
    #                                                 ) (x)
    #                         noise_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
    #                                                         activation=layer_config['act'], 
    #                                                         use_bias=config['bias_dense']), 
    #                                                     name='noise_hat'
    #                                                 ) (x)
    #                         if config['bn']:
    #                             piano_hat = BatchNormalization() (piano_hat)
    #                             noise_hat = BatchNormalization() (noise_hat)
                            
    #                         piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
    #                         noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
                    
    #                     else:
    #                         piano_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
    #                                                         activation=layer_config['act'], 
    #                                                         use_bias=config['bias_dense']), 
    #                                                     name='piano_hat'
    #                                                 ) (x)
    #                         noise_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
    #                                                         activation=layer_config['act'], 
    #                                                         use_bias=config['bias_dense']),
    #                                                     name='noise_hat'
    #                                                 ) (x)
    #                         if config['bn']:
    #                             piano_hat = BatchNormalization() (piano_hat)
    #                             noise_hat = BatchNormalization() (noise_hat)

    #                 else:
    #                     x = TimeDistributed(Dense(features // layer_config['nrn_div'],
    #                                             activation=layer_config['act'], 
    #                                             use_bias=config['bias_dense']), 
    #                                     ) (input_layer if (i == 0 and not config['scale']) else x)
    #                     if config['bn']:
    #                         x = BatchNormalization() (x)

    #             prev_layer_type = curr_layer_type
        
    #     elif test > 0:
    #         if test == 1:
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         # The difference in mem use between test 1 @ 2 is much an long a rnn takes
    #         if test == 2:
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         if test == 3:
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         if test == 4:
    #             x = LSTM(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = LSTM(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x)
    #             x = LSTM(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         # The difference in mem use between test 2 @ 4 is how much more an lstm takes
    #         if test == 5:
    #             x = LSTM(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = LSTM(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         if test == 6:
    #             x = LSTM(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer)
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         if test == 7:
    #             x = Standardize(t_mean, t_std) (input_layer)
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #             noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
    #         # The difference in mem use between test 6 & 2 is what doubling dim red does to mem usage
    #         if test == 8:
    #             x = Standardize(t_mean, t_std) (input_layer)
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #             noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
    #         if test == 9:
    #             x = Standardize(t_mean, t_std) (input_layer)
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #             noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
    #         if test == 10:
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x)
    #             x = Concatenate() ([x, input_layer])
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         # The difference in mem use between test 1 @ 2 is much an long a rnn takes
    #         if test == 11:
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             x = Concatenate() ([x, input_layer])
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         if test == 12:
    #             x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = Concatenate() ([x, input_layer])
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

    #         if test == 13:
    #             x = Bidirectional(SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True)) (input_layer) 
    #             x = Bidirectional(SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True)) (x) 
    #             x = Bidirectional(SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True)) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         # The difference in mem use between test 1 @ 2 is much an long a rnn takes
    #         if test == 14:
    #             x = Bidirectional(SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True)) (input_layer) 
    #             x = Bidirectional(SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True)) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         if test == 15:
    #             x = Bidirectional(SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True)) (input_layer) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

    #         if test == 16:
    #             x = SimpleRNN(features - 1, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = SimpleRNN(features - 1, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             x = SimpleRNN(features - 1, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         # The difference in mem use between test 6 & 2 is what doubling dim red does to mem usage
    #         if test == 17:
    #             x = SimpleRNN(features - 1, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             x = SimpleRNN(features - 1, 
    #                     activation='relu', 
    #                     return_sequences=True) (x) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #         if test == 18:
    #             x = SimpleRNN(features - 1, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #             piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #             noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

    #     # Use pre-configurations (default)
    #     else:
    #         x = SimpleRNN(features // 2, 
    #                     activation='relu', 
    #                     return_sequences=True) (input_layer) 
    #         x = SimpleRNN(features // 2, 
    #                 activation='relu',
    #                 return_sequences=True) (x)
    #         piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
    #         noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    #     piano_pred = TimeFreqMasking(epsilon=epsilon, 
    #                                 name='piano_pred') ((piano_hat, noise_hat, input_layer))
    #     noise_pred = TimeFreqMasking(epsilon=epsilon, 
    #                                 name='noise_pred') ((noise_hat, piano_hat, input_layer))

    #     # model = Model(inputs=input_layer, outputs=[piano_pred, noise_pred])

    #     # # Keras debug block
    #     # debug_piano_model = Model(
    #     #     inputs=model.inputs,
    #     #     # inputs=model.layers[3].output,
    #     #     # outputs=[model.layers[0].output] + model.outputs,
    #     #     outputs=[model.layers[2].output, model.layers[3].output, model.layers[5].output],
    #     #     name='Debug Piano Model (rnn2 out -> piano_hat out -> piano_pred out)'
    #     # )
    #     # debug_noise_model = Model(
    #     #     inputs=model.inputs,
    #     #     outputs=[model.layers[2].output, model.layers[4].output, model.layers[6].output],
    #     #     name='Debug Noise Model (rnn2 out -> noise_hat out -> noise_pred out)'
    #     # )
    #     # xs = tf.random.normal((3, sequences, features))
    #     # # print('DEBUG Piano Model Summary:')
    #     # # print(debug_piano_model.summary())
    #     # print('DEBUG Piano Model Run:')
    #     # # print(debug_piano_model(xs, training=True))

    #     # debug_piano_model_outputs = debug_piano_model(xs, training=True)
    #     # rnn_o, dense_o, mask_o = debug_piano_model_outputs[0].numpy(), debug_piano_model_outputs[1].numpy(), debug_piano_model_outputs[2].numpy()
    #     # print('Shape rnn out:', rnn_o.shape)
    #     # print('Shape dense out:', dense_o.shape)
    #     # print('Shape mask out:', mask_o.shape)
    #     # # print('Inf in rnn out:', True in np.isinf(rnn_o))
    #     # # print('Inf in dense out:', True in np.isinf(dense_o))
    #     # # print('Inf in mask out:', True in np.isinf(mask_o))
    #     # # print('NaN in rnn out:', True in np.isnan(rnn_o))
    #     # # print('NaN in dense out:', True in np.isnan(dense_o))
    #     # # print('NaN in mask out:', True in np.isnan(mask_o))
    #     # print()

    #     # # print('DEBUG Noise Model Summary:')
    #     # # print(debug_noise_model.summary())
    #     # print('DEBUG Noise Model Run:')
    #     # # print(debug_noise_model(xs, training=True))
    #     # debug_noise_model_outputs = debug_noise_model(xs, training=True)
    #     # rnn_o, dense_o, mask_o = debug_noise_model_outputs[0].numpy(), debug_noise_model_outputs[1].numpy(), debug_noise_model_outputs[2].numpy()
    #     # print('Shape rnn out:', rnn_o.shape)
    #     # print('Shape dense out:', dense_o.shape)
    #     # print('Shape mask out:', mask_o.shape)
    #     # # print('Inf in rnn out:', True in np.isinf(rnn_o))
    #     # # print('Inf in dense out:', True in np.isinf(dense_o))
    #     # # print('Inf in mask out:', True in np.isinf(mask_o))
    #     # # print('NaN in rnn out:', True in np.isnan(rnn_o))
    #     # # print('NaN in dense out:', True in np.isnan(dense_o))
    #     # # print('NaN in mask out:', True in np.isnan(mask_o))
    #     # print()
    #     # # print('Model Layers:')
    #     # # print([layer.name for layer in model.layers])
    #     # # ['piano_noise_mixed', 'simple_rnn', 'simple_rnn_1', 'piano_hat', 'noise_hat', 'piano_pred', 'noise_pred']    
        
    #     # disc_loss = None
    #     if pre_trained_wgts is not None:
    #             # loss_const_tensor = tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
    #             preds_and_gamma = Concatenate(axis=0) ([piano_pred, 
    #                                                 noise_pred, 
    #                                                 #  loss_const_tensor
    #                                                 tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
    #                                                 ])
    #             # MIXED PRECISION - not sure if test case is necessary
    #             # if not pc_run:
    #             #     preds_and_gamma = Activation('linear', name='mp_output', dtype='float32') (preds_and_gamma)

    #             model = Model(inputs=input_layer, outputs=preds_and_gamma)

    #             print('Only loading pre-trained weights for prediction')
    #             model.set_weights(pre_trained_wgts)
    #     elif keras_fit:
    #         # loss_const_tensor = tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
    #         preds_and_gamma = Concatenate(axis=0) ([piano_pred, 
    #                                             noise_pred, 
    #                                             #  loss_const_tensor
    #                                             tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
    #                                             ])
    #         # MIXED PRECISION
    #         # if not pc_run:
    #         #     preds_and_gamma = Activation('linear', name='mp_output', dtype='float32') (preds_and_gamma)

    #         model = Model(inputs=input_layer, outputs=preds_and_gamma)
    #         # model = Model(inputs=input_layer, outputs=output)

    #         # Combine piano_pred, noise_pred & loss_const into models output!
    #         # loss_const_tensor = tf.reshape(tf.constant(loss_const), [None, sequences, 1])
    #         # output = Concatenate() ([piano_pred, noise_pred, loss_const_tensor])

    #         model.compile(optimizer=optimizer, loss=discrim_loss)

    #     return model    #, disc_loss

    # If use preprocess_data_generator, need to pass in data_path and pad_len

    train_generator = data.fixed_data_generator(
            x_train_files, y1_train_files, y2_train_files, num_train,
            batch_size=batch_size, num_seq=n_seq, num_feat=n_feat, pc_run=pc_run, 
            dmged_piano_artificial_noise=dataset2)
    validation_generator = data.fixed_data_generator(
            x_val_files, y1_val_files, y2_val_files, num_val,
            batch_size=batch_size, num_seq=n_seq, num_feat=n_feat, pc_run=pc_run, 
            dmged_piano_artificial_noise=dataset2)

    train_dataset = tf.data.Dataset.from_generator(
        data.make_gen_callable(train_generator), 
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, n_seq, n_feat), (None, n_seq, n_feat*2)),
    )
    val_dataset = tf.data.Dataset.from_generator(
        data.make_gen_callable(validation_generator), 
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, n_seq, n_feat), (None, n_seq, n_feat*2)),
    )

    train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
   
    print('Making model...')
    model = make_model(n_feat, n_seq, name='Training Model', epsilon=epsilon, loss_const=loss_const,
                            config=config, t_mean=t_mean, t_std=t_std, optimizer=optimizer,
                            pc_run=pc_run)
    print(model.summary())

    print('Going into training now...')
    # log_dir = '../logs/keras_fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    hist = model.fit(train_dataset,
                    steps_per_epoch=math.ceil(num_train / batch_size),
                    epochs=epochs,
                    validation_data=val_dataset,
                    validation_steps=math.ceil(num_val / batch_size),
                    callbacks=[EarlyStopping('val_loss', patience=patience, mode='min')])#,
                            # Done memory profiling
                            # TensorBoard(log_dir=log_dir, profile_batch='2, 4')])   # 10' # by default, profiles 2nd batch
    history = hist.history
    # Need to install additional unnecessary libs
    # if not pc_run and grid_search_iter is None:
    #     tf.keras.utils.plot_model(model, 
    #                               (gs_path + 'model' + str(grid_search_iter) + 'of' + str(combos) + '.png'
    #                               if grid_search_iter is not None else
    #                               'last_trained_model.png'), 
    #                               show_shapes=True)
 
    pc_run_str = '' if pc_run else '_noPC'
    if pc_run or grid_search_iter is None:
        #  Can't for imperative models
        model.save(recent_model_path)

        # print('History Dictionary Keys:', hist.history.keys())
        # 'val_loss', 'loss'
        print('Val Loss:\n', history['val_loss'])
        print('Loss:\n', history['loss'])

        epoch_r = range(1, len(history['loss'])+1)
        plt.plot(epoch_r, history['val_loss'], 'b', label = 'Validation Loss')
        plt.plot(epoch_r, history['loss'], 'bo', label = 'Training Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('brahms_restore_ml/drnn/train_val_loss_chart' + pc_run_str + '.png')

    if ret_queue is not None:
        # ret_queue not a queue, it's send end of pipe
        ret_queue.send((history['loss'], history['val_loss']))
    
    else:
        return model


def get_hp_configs(bare_config_path, pc_run=False):
    # import tensorflow as tf

    # IMPORTANT: 1st GS - GO FOR WIDE RANGE OF OPTIONS & LESS OPTIONS PER HP
    # TEST FLOAT16 - double batch size
    # MIXED PRECISION   - double batch size (can't on PC still b/c OOM), for V100: multiple of 8
    # batch_size_optns = [3] if pc_run else [8] # [16, 24] OOM on f35 w/ old addloss model
    # OOM BOUND TEST
    # batch_size_optns = [3] if pc_run else [12, 18]  
    # batch_size_optns = [5] if pc_run else [8, 16]    # OOM on f35, and on PC, BUT have restart script now
    # 11/19/20 for PC - too late, just run this over break - test SGD vs mini-batch SGD (memory conservative)
    # batch_size_optns = [1, 3] if pc_run else [4, 8]    # OOM on f35 and on PC, w/ restart script,
    batch_size_optns = [1, 3] if pc_run else [8, 16]    # Fix TF mem management w/ multiprocessing - it lets go of mem after a model train now

    # # MEM BOUND TEST
    # batch_size_optns = [8] # - time
    # # batch_size_optns = [25]

    # batch_size_optns = [5] if pc_run else [8, 12] 
    # epochs total options 10, 50, 100, but keep low b/c can go more if neccesary later (early stop pattern = 5)
    epochs_optns = [10]
    # loss_const total options 0 - 0.3 by steps of 0.05
    # loss_const_optns = [0.05, 0.2]
    # loss_const_optns = [0.05, 0.1] if pc_run else [0.05]    # first of two HPs dropping, PC GS time constraint
    loss_const_optns = [0.05, 0.1] if pc_run else [0.05, 0.1]    # Multi-processing fix -> orig numbers

    # Optimizers ... test out Adaptive Learning Rate Optimizers (RMSprop & Adam) Adam ~ RMSprop w/ momentum
    # Balance between gradient clipping and lr for exploding gradient
    # If time permits, later grid searches explore learning rate & momentum to fine tune
    
    # OLD
    # WORKED!!!! (very low lr - 2 orders of mag lower than default) at 9:45 am checked output
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=10, learning_rate=0.00001) # Random HP
    # Try next?
    # ALMOST worked, bcame NaN at end, so bad result
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=1, learning_rate=0.0001) # Random HP
    # Failed
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=0.5)
    # Failed
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=1)
    # ALMOST
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001) # Random HP
    # Find optimal balance betwwen clipval & lr for random HPs
    # ALMOST worked, NaN at end
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=10, learning_rate=0.0001) # Random HP
    # (very low clipvalue - 2/3 orders of mag higher than default ~1/0.1)
    # Failed
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=100, learning_rate=0.0001) # Random HP
    # Is learning rate only thing that matters? YES or does clip val help no
    # ALMOST, but became NaN earlier - lr is more effective
    #   When clip calue is too high w/ lr -> bad, else almost works
    #       does work when learning rate = 0.00001
    # train_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001) # Random HP
    
    # Optimizers, should non-PC have more options b/c LSTM could help w/ expl gradi
    #   TODO Test - on PC w/ LSTM to find out after this

    # Trials for good results
    # Does clipvalue do anything by itslef (cv=100)? - no  10? - yes, 19 million
    # What's the ghighest learning rate by itself that makes it work? 
    #   0.0001 (R & A)->0.0005 (not R & not A) -> 0.0008 no
    #   11 million                                            
    # If we gradclip with a a little higher learning rate, still work? 
    #   cv=10 & lr=0.0005 yes, cv=10 & lr=0.001 no, cv=5 & lr=0.0008 yes, cv=5 & lr=0.001 no, cv=10 & lr=0.0008 no
    #   5 million (RMSprop)                         14 million val loss                         
    # optimizer_optns = [
    #                   (tf.keras.optimizers.RMSprop(clipvalue=10), 10, 0.001, 'RMSprop'),
    #                   (tf.keras.optimizers.Adam(clipvalue=10), 10, 0.001, 'Adam')
    #                   ]
    # FYI - Best seen: clipvalue=10, lr=0.0005 - keep in mind for 2nd honed-in GS
    # # FLOAT16
    # optimizer_optns = [
    #                   (tf.keras.optimizers.RMSprop(learning_rate=0.0001), -1, 0.0001, 'RMSprop'),
    #                   (tf.keras.optimizers.RMSprop(clipvalue=10), 10, 0.001, 'RMSprop'),
    #                   (tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-4), -1, 0.0001, 'Adam'),
    #                   (tf.keras.optimizers.Adam(clipvalue=10, epsilon=1e-4), 10, 0.001, 'Adam')
    #                   ]
    # MIXED PRECISION - doesn't support gradient clipping or specifically clipvalue
    # FOR TIME CONSTRAINT
    # if pc_run:
    optimizer_optns = [(None, 0.0001, 'RMSprop'), 
                       (10, 0.001, 'RMSprop'),
                       (None, 0.0001, 'Adam'), 
                       (10, 0.001, 'Adam')]
    # else:
    #     optimizer_optns = [
    #                     (tf.keras.optimizers.RMSprop(learning_rate=0.0001), -1, 0.0001, 'RMSprop'),
    #                     (tf.keras.optimizers.RMSprop(clipvalue=10), 10, 0.001, 'RMSprop'),
    #                     # (tf.keras.optimizers.RMSprop(clipvalue=1), 1, 0.001, 'RMSprop'),
    #                     (tf.keras.optimizers.Adam(learning_rate=0.0001), -1, 0.0001, 'Adam'),
    #                     (tf.keras.optimizers.Adam(clipvalue=10), 10, 0.001, 'Adam'),
    #                     # (tf.keras.optimizers.Adam(clipvalue=1), 1, 0.001, 'Adam')
    #                     ]

    train_configs = {'batch_size': batch_size_optns, 'epochs': epochs_optns,
                     'loss_const': loss_const_optns, 'optimizer': optimizer_optns}
    
    # # REPL TEST - arch config, all config, optiizer config
    # dropout_optns = [(0.0,0.0)]    # For RNN only    IF NEEDED CAN GO DOWN TO 2 (conservative value)
    # scale_optns = [False]
    # rnn_skip_optns = [False]
    # bias_rnn_optns = [True]     # False
    # bias_dense_optns = [True]   # False
    # bidir_optns = [True]
    # bn_optns = [False]                    # For Dense only
    # # TEST - failed - OOM on PC
    # # rnn_optns = ['LSTM'] if pc_run else ['RNN', 'LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model              
    # rnn_optns = ['RNN'] if pc_run else ['RNN', 'LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model
    # if pc_run:
    #     # TEST PC
    #     # with open(bare_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
    #     with open(bare_config_path + 'hp_arch_config_final.json') as hp_file:
    #         bare_config_optns = [json.load(hp_file)['archs'][3]]
    # else:
    #     # with open(bare_config_path + 'hp_arch_config_largedim.json') as hp_file:
    #     with open(bare_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
    #         bare_config_optns = json.load(hp_file)['archs']

    # IMPORTANT: Order arch config options by mem-intensive HPs first,
    #           to protect against OOM on grid search
    # dropout_optns = [(0.0,0.0), (0.2,0.2), (0.2,0.5), (0.5,0.2), (0.5,0.5)]   # For RNN only
    # # MEM BOUND TEST
    # dropout_optns = [(0.25,0.25)]
    dropout_optns = [(0.0,0.0), (0.25,0.25)]    # For RNN only    IF NEEDED CAN GO DOWN TO 2 (conservative value)
    # # MEM BOUND TEST
    # scale_optns = [True]
    scale_optns = [False, True]
    # # MEM BOUND TEST
    # rnn_skip_optns = [True]
    rnn_skip_optns = [False, True]
    bias_rnn_optns = [True]     # False
    bias_dense_optns = [True]   # False
    # HP range test - only True
    # # MEM BOUND TEST
    # bidir_optns = [True]
    bidir_optns = [False, True]
    # # MEM BOUND TEST
    # bn_optns = [True]  
    bn_optns = [False, True]                    # For Dense only
    # # MEM BOUND TEST
    # rnn_optns = ['RNN']
    rnn_optns = ['RNN']
    # rnn_optns = ['RNN'] if pc_run else ['RNN', 'LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model  
    # MIXED PRECISION combat NaNs            
    # rnn_optns = ['RNN'] if pc_run else ['LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model
    # rnn_optns = ['RNN'] # F35 OOM w/ mixed precision BUT batch size too high?
    if pc_run:
        # TEST PC
        # with open(bare_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
        with open(bare_config_path + 'hp_arch_config_final.json') as hp_file:
            bare_config_optns = json.load(hp_file)['archs']
    else:
        # with open(bare_config_path + 'hp_arch_config_largedim.json') as hp_file:
        # with open(bare_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
        with open(bare_config_path + 'hp_arch_config_final_no_pc_long.json') as hp_file:
            bare_config_optns = json.load(hp_file)['archs']
    
    # # MEM BOUND TEST
    # bare_config_optns = [bare_config_optns[-1]]
    arch_config_optns = []
    for config in bare_config_optns:  
        for rnn_optn in rnn_optns:
            for bidir_optn in bidir_optns:
                for scale_optn in scale_optns:
                    for bn_optn in bn_optns:  
                        for rnn_skip_optn in rnn_skip_optns:
                            for bias_rnn_optn in bias_rnn_optns:
                                for bias_dense_optn in bias_dense_optns:
                                    for dropout_optn in dropout_optns:   
                                        # Make a unique copy for each factor combo
                                        curr_config = deepcopy(config)  
                                        curr_config['scale'] = scale_optn
                                        curr_config['rnn_res_cntn'] = rnn_skip_optn
                                        curr_config['bias_rnn'] = bias_rnn_optn
                                        curr_config['bias_dense'] = bias_dense_optn
                                        curr_config['bidir'] = bidir_optn
                                        curr_config['rnn_dropout'] = dropout_optn
                                        curr_config['bn'] = bn_optn
                                        if rnn_optn == 'LSTM':
                                            for i, layer in enumerate(config['layers']):
                                                if layer['type'] == 'RNN':
                                                    curr_config['layers'][i]['type'] = rnn_optn
                                        arch_config_optns.append(curr_config) 
 
    return train_configs, arch_config_optns

# GRID SEARCH FUNCTION
def grid_search(x_train_files, y1_train_files, y2_train_files, 
                x_val_files, y1_val_files, y2_val_files,
                n_feat, n_seq, 
                epsilon, 
                t_mean, t_std,
                train_configs, arch_config_optns,
                gsres_path, early_stop_pat=3, pc_run=False, 
                gs_id='', restart=False, dataset2=False):

    # IMPORTANT to take advantage of what's known in test data to minimize factors
    # Factors: batchsize, epochs, loss_const, optimizers, gradient clipping,
    # learning rate, lstm/rnn, layer type/arch, tanh activation, num neurons in layer,
    # bidiric layer, amp var aug range, batch norm, skip connection over lstms,
    # standardize input & un-standardize output  
    # Maybe factor? Dmged/non-dmged piano input 
    print('\nPC RUN:', pc_run, '\n\nGRID SEARCH ID:', gs_id if len(gs_id) > 0 else 'N/A', '\n')

    num_train, num_val = len(y1_train_files), len(y1_val_files)

    batch_size_optns = train_configs['batch_size']
    epochs_optns = train_configs['epochs']
    loss_const_optns = train_configs['loss_const']
    optimizer_optns = train_configs['optimizer']

    combos = (len(batch_size_optns) * len(epochs_optns) * len(loss_const_optns) *
              len(optimizer_optns) * len(arch_config_optns))
    print('\nGS COMBOS:', combos, '\n')

    # Start where last left off, if applicable:
    if not restart:
        gs_iters_so_far = []
        # Search through grid search output directory
        base_dir = os.getcwd()
        os.chdir(gsres_path)
        if len(gs_id) > 0:
            gs_result_files = [f_name for f_name in os.listdir(os.getcwd()) if 
                               (f_name.endswith('txt') and f_name[0].isdigit() and f_name[0] == gs_id)]

            for f_name in gs_result_files:
                gs_iter = [int(token) for token in f_name.split('_') if token.isdigit()][1]  
                gs_iters_so_far.append(gs_iter)

        else:
            gs_result_files = [f_name for f_name in os.listdir(os.getcwd()) if 
                               (f_name.endswith('txt') and f_name[0] == 'r')]

            for f_name in gs_result_files:
                gs_iter = [int(token) for token in f_name.split('_') if token.isdigit()][0]  
                gs_iters_so_far.append(gs_iter)

        os.chdir(base_dir)
        # Know the last done job, if any jobs were done
        gs_iters_so_far.sort(reverse=True)
        last_done = gs_iters_so_far[0] if (len(gs_iters_so_far) > 0) else 0
        
        if last_done > 0:
            print('RESUMING GRID SEARCH AT ITERATION', last_done + 1, '\n')

    # Format grid search ID for filenames:
    if len(gs_id) > 0:
        gs_id += '_'

    # IMPORTANT: Grab HPs in order of mem-instensiveness, reduces chances OOM on grid search
    # Full grid search loop
    grid_results_val, grid_results, gs_iter = {}, {}, 1
    for batch_size in batch_size_optns:     # Batch size is tested first -> fast OOM-handling iterations
        for arch_config in arch_config_optns:
            for epochs in epochs_optns:
                for loss_const in loss_const_optns:
                    for clip_val, lr, opt_name in optimizer_optns:
                        
                        if restart or (gs_iter > last_done):

                            print('BEGINNING GRID-SEARCH ITER', (str(gs_iter) + '/' + str(combos) + ':\n'), 
                                'batch_size:', batch_size, 'epochs:', epochs, 'loss_const:', loss_const,
                                'optimizer:', opt_name, 'clipvalue:', clip_val, 'learn_rate:', lr, 
                                '\narch_config', arch_config, '\n')

                            send_end, recv_end = multiprocessing.Pipe()
                            # Multi-processing hack - make TF let go of mem after a model creation & training
                            process_train = multiprocessing.Process(target=evaluate_source_sep, args=(
                                                                    x_train_files, y1_train_files, y2_train_files, 
                                                                    x_val_files, y1_val_files, y2_val_files,
                                                                    num_train, num_val,
                                                                    n_feat, n_seq, 
                                                                    batch_size, loss_const,
                                                                    epochs, 
                                                                    opt_name, clip_val, lr,
                                                                    early_stop_pat,
                                                                    epsilon,
                                                                    arch_config, None, pc_run,
                                                                    t_mean, t_std,
                                                                    gs_iter,
                                                                    gsres_path,
                                                                    combos, gs_id,
                                                                    send_end, dataset2))

                            process_train.start()
                    
                            # Keep polling until child errors or child success (either one guaranteed to happen)
                            losses, val_losses = None, None
                            while process_train.is_alive():
                                time.sleep(60)
                                if recv_end.poll():
                                    losses, val_losses = recv_end.recv()
                                    break

                            if losses == None or val_losses == None:
                                print('\nERROR happened in child and it died')
                                exit(1)

                            print('Losses from pipe:', losses)
                            print('Val. losses from pipe:', val_losses)
                            process_train.join()

                            # Do multiple runs of eval_src_sep to avg over randomness?
                            curr_basic_loss = {'batch_size': batch_size, 
                                                'epochs': epochs, 'gamma': loss_const,
                                                'optimizer': opt_name, 'clip value': clip_val,
                                                'learning rate': lr, 'all_loss': losses}
                            curr_basic_val_loss = {'batch_size': batch_size, 
                                                'epochs': epochs, 'gamma': loss_const,
                                                'optimizer': opt_name, 'clip value': clip_val,
                                                'learning rate': lr, 'all_loss': val_losses}
                            # Write results to file
                            pc_run_str = '' if pc_run else '_noPC'
                            with open(gsres_path + gs_id + 'result_' + str(gs_iter) + '_of_' + str(combos) + pc_run_str + '.txt', 'w') as w_fp:
                                w_fp.write(str(val_losses[-1]) + '\n')
                                w_fp.write('VAL LOSS ^\n')
                                w_fp.write(str(losses[-1]) + '\n')
                                w_fp.write('LOSS ^\n')
                                w_fp.write(json.dumps({**arch_config, **curr_basic_val_loss}) + '\n')
                                w_fp.write('VAL LOSS FACTORS ^\n')
                                w_fp.write(json.dumps({**arch_config, **curr_basic_loss}) + '\n')
                                w_fp.write('LOSS FACTORS^\n')

                        gs_iter += 1

# https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
# Only want to predict one batch = 1 song, solutions:
#   - Online learning (train one batch/song at a time/epoch) - possible instability in training
#   - Predict all batches at once (impossible - only one orig Brahms)
#   - CHOSE Copy weights from fit network to a newly created network

# MODEL INFERENCE FUNCTION
def infer(x, phases, wdw_size, model, loss_const, optimizer, 
          seq_len, n_feat, batch_size, epsilon, output_path, sr, orig_sig_type,
          config=None, t_mean=None, t_std=None, pc_run=False, name_addon=''):
    
    # from .model import make_model
    import tensorflow as tf

    # Must make new model, b/c Brahms spgm has different num timesteps, or pad Brahms spgm
    orig_sgmts = x.shape[0]
    # print('x shape:', x.shape)
    # print('model 1st layer input shape:', model.layers[0].input_shape[1])
    # print('model 1st layer output shape:', model.layers[0].output_shape[1])
    deficit = model.layers[0].input_shape[0][1] - x.shape[0]
    x = np.concatenate((x, np.zeros((deficit, x.shape[1]))))
    x = np.expand_dims(x, axis=0)   # Give a samples dimension (1 sample)
    print('x shape to be predicted on (padded) (w/ a batch dimension):', x.shape)

    # print('Inference Model:')
    # model = make_model(n_feat, seq_len, loss_const=loss_const, optimizer=optimizer,
    #                    pre_trained_wgts=model.get_weights(), name='Inference Model',
    #                    epsilon=epsilon, config=config, t_mean=t_mean, t_std=t_std,
    #                    pc_run=pc_run)
    # print(model.summary())
    # For small amts of input that fit in one batch: __call__ > predict - didn't work :/
    # clear_spgm, noise_spgm = model([x, x, x], batch_size=batch_size, training=False)
    result_spgms = model.predict(x, batch_size=batch_size)
    clear_spgm, noise_spgm = tf.split(result_spgms[:-1, :, :], num_or_size_splits=2, axis=0)
    clear_spgm = clear_spgm.numpy().reshape(-1, n_feat)
    noise_spgm = noise_spgm.numpy().reshape(-1, n_feat)
    clear_spgm, noise_spgm = clear_spgm[:orig_sgmts], noise_spgm[:orig_sgmts]

    if pc_run:
        plot_matrix(clear_spgm, name='clear_output_spgm', ylabel='Frequency (Hz)', 
                ratio=SPGM_BRAHMS_RATIO)
        plot_matrix(noise_spgm, name='noise_output_spgm', ylabel='Frequency (Hz)', 
                ratio=SPGM_BRAHMS_RATIO)

    synthetic_sig = make_synthetic_signal(clear_spgm, phases, wdw_size, 
                                          orig_sig_type, ova=True, debug=False)
    wavfile.write(output_path + 'restore' + name_addon + '.wav', sr, synthetic_sig)

    synthetic_sig = make_synthetic_signal(noise_spgm, phases, wdw_size, 
                                          orig_sig_type, ova=True, debug=False)
    wavfile.write(output_path + 'noise' + name_addon + '.wav', sr, synthetic_sig)


# BRAHMS RESTORATION FUNCTION (USES INFERENCE)
def restore_audio_file(output_path, recent_model_path, wdw_size, epsilon, loss_const, 
                       opt_name, opt_clip_val, opt_lr,
                       test_filepath=None, test_sig=None, test_sr=None, 
                       config=None, t_mean=None, t_std=None, pc_run=False, name_addon=''):
    import tensorflow as tf

    infer_model = tf.keras.models.load_model(recent_model_path, compile=False)
    print('Inference Model:')
    print(infer_model.summary())
    # Instantiate optimizer
    optimizer = (tf.keras.optimizers.RMSprop(clipvalue=opt_clip_val, learning_rate=opt_lr) if 
                    opt_name == 'RMSprop' else
                tf.keras.optimizers.Adam(clipvalue=opt_clip_val, learning_rate=opt_lr))
    
    if test_filepath:
        # Load in testing data - only use sr of test
        print('Restoring audio of file:', test_filepath)
        test_sr, test_sig = wavfile.read(test_filepath)
    test_sig_type = test_sig.dtype

    # Spectrogram creation - test. Only use phases of test
    test_spgm, test_phases = make_spectrogram(test_sig, wdw_size, epsilon, ova=True, debug=False)
    test_feat = test_spgm.shape[1]
    test_seq = test_spgm.shape[0]
    test_batch_size = 1

    infer(test_spgm, test_phases, wdw_size, infer_model, loss_const=loss_const, optimizer=optimizer,
        seq_len=test_seq, n_feat=test_feat, batch_size=test_batch_size, epsilon=epsilon,
        output_path=output_path, sr=test_sr, orig_sig_type=test_sig_type,
        config=config, t_mean=t_mean, t_std=t_std, pc_run=pc_run, name_addon=name_addon)