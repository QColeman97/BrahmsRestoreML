import numpy as np
import datetime
import math
import random
import json
import os
import sys
import re
from copy import deepcopy
import multiprocessing

import tensorflow as tf


def fixed_data_generator(#x_files, y1_files, y2_files, 
                        num_samples, batch_size, num_seq, num_feat, pc_run, 
                         dmged_piano_artificial_noise=False, pad_len=-1, wdw_size=4096, epsilon=10 ** (-10)):
    while True: # Loop forever so the generator never terminates
        # for i in range(num_samples):
        for offset in range(0, num_samples, batch_size):
            # x_batch_labels = x_files[offset:offset+batch_size]
            # y1_batch_labels = y1_files[offset:offset+batch_size]
            # y2_batch_labels = y2_files[offset:offset+batch_size]
            # if (num_samples / batch_size == 0):
                # TEST FLOAT16
                # MIXED PRECISION - hail mary try

            # actual_batch_size = batch_size
            x, y1, y2 = (np.empty((batch_size, num_seq, num_feat)).astype('float32'),
                        np.empty((batch_size, num_seq, num_feat)).astype('float32'),
                        np.empty((batch_size, num_seq, num_feat)).astype('float32'))
            # else:
            #     actual_batch_size = len(x_batch_labels)
            #     x, y1, y2 = (np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'),
            #                 np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'),
            #                 np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'))
            
            # for i in range(actual_batch_size):
            #     pn_filepath = x_batch_labels[i]
            #     pl_filepath = y1_batch_labels[i]
            #     nl_filepath = y2_batch_labels[i]

            #     # MIXED PRECISION - hail mary try
            #     noise_piano_spgm = np.load(pn_filepath)# .astype('float32')
            #     piano_label_spgm = np.load(pl_filepath)# .astype('float32')
            #     noise_label_spgm = np.load(nl_filepath)# .astype('float32')

            #     x[i] = noise_piano_spgm
            #     y1[i] = piano_label_spgm
            #     y2[i] = noise_label_spgm
            
            yield ([x, np.concatenate((y1, y2), axis=-1)])


def test_func(i, send_end,
                        train_generator, validation_generator,
                        # train_step_func, test_step_func,
                        num_train, num_val, n_feat, n_seq, batch_size, 
                        loss_const, epochs=20, 
                        optimizer=None,
                        # optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.75),
                        patience=100, epsilon=10 ** (-10), config=None, recent_model_path=None, pc_run=False,
                        t_mean=None, t_std=None, grid_search_iter=None, gs_path=None, combos=None, gs_id='',
                        keras_fit=False, ret_queue=None):

    import tensorflow as tf
    # from tensorflow import keras
    # from tensorflow.keras.losses import Loss
    from tensorflow.keras.layers import Input, SimpleRNN, Dense, Lambda, TimeDistributed, Layer, LSTM, Bidirectional, BatchNormalization, Concatenate, Activation
    from tensorflow.keras.models import Model
    # from tensorflow.keras.utils import Sequence
    # from tensorflow.keras.activations import relu
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

    pc_run = False
    keras_fit = True

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
            choice = random.randint(0, 9)
            chosen_gpu = random.randint(0, len(physical_gpus)-1) if (choice == 0) else 0
            print("Restricting TF run to only use 1 GPU:", physical_gpus[chosen_gpu], "\n")
            tf.config.experimental.set_visible_devices(physical_gpus[chosen_gpu], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    class Standardize(Layer):
        def __init__(self, mean, std, **kwargs):
            super(Standardize, self).__init__(**kwargs)
            self.mean = mean
            self.std = std

        def call(self, input):
            input -= self.mean
            input /= self.std
            return input


    class UnStandardize(Layer):
        def __init__(self, mean, std, **kwargs):
            super(UnStandardize, self).__init__(**kwargs)
            self.mean = mean
            self.std = std

        def call(self, input):
            input *= self.std
            input += self.mean
            return input


    # TF Masking layer has too compilcated operations for a lambda, and want to serialize model
    class TimeFreqMasking(Layer):

        # Init is for input-independent variables
        # def __init__(self, piano_flag, **kwargs):
        def __init__(self, epsilon, **kwargs):
            # MAKE LAYER DEAL IN FLOAT16
            # TEST FLOAT16
            # kwargs['autocast'] = False
            # MIXED PRECISION - output layer needs to produce float32
            # kwargs['dtype'] = 'float32' # - or actually try in __init__ below
            # super(TimeFreqMasking, self).__init__(dtype='float32', **kwargs)
            super(TimeFreqMasking, self).__init__(**kwargs)
            # self.piano_flag = piano_flag
            self.epsilon = epsilon

        # No build method, b/c passing in multiple inputs to layer (no single shape)

        def call(self, inputs):
            # Try this alternative format if below doesn't work
            # self.total = tf.Variable(initial_value=y_hat_other, trainable=False)
            # self.total.assign_add(tf.reduce_sum(inputs, axis=0))
            # return self.total

            # y_hat_self, y_hat_other, x_mixed = inputs[0], inputs[1], inputs[2]
            y_hat_self, y_hat_other, x_mixed = inputs

            # print('TYPES IN TF MASKING:', y_hat_self.dtype, y_hat_other.dtype, x_mixed.dtype)

            mask = tf.abs(y_hat_self) / (tf.abs(y_hat_self) + tf.abs(y_hat_other) + self.epsilon)
            # print('Mask Shape:', mask.shape)
            # ones = tf.convert_to_tensor(np.ones(mask.shape).astype('float32'))
            # print('Ones Shape:', ones.shape)
            # y_tilde_self = mask * x_mixed if (self.piano_flag) else (ones - mask) * x_mixed
            y_tilde_self = mask * x_mixed

            # print('Y Tilde Shape:', y_tilde_self.shape)
            return y_tilde_self
        
        # config only contains things in __init__
        def get_config(self):
            config = super(TimeFreqMasking, self).get_config()
            config.update({'epsilon': self.epsilon})
            return config
        
        def from_config(cls, config):
            return cls(**config)


    def discrim_loss(y_true, y_pred):
        piano_true, noise_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
        loss_const = y_pred[-1, :, :][0][0]
        piano_pred, noise_pred = tf.split(y_pred[:-1, :, :], num_or_size_splits=2, axis=0)

        last_dim = piano_pred.shape[1] * piano_pred.shape[2]
        return (
            tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
            (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
            tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) -
            (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1))
        )

    def make_model(features, sequences, name='Model', epsilon=10 ** (-10),
                        loss_const=0.05, config=None, t_mean=None, t_std=None, 
                        optimizer=tf.keras.optimizers.RMSprop(),
                        pre_trained_wgts=None,
                        # GPU mem as func of HP TEST
                        # test=16, 
                        test=0, 
                        pc_run=False,
                        keras_fit=False):
        # TEST FLOAT16
        # MIXED PRECISION
        input_layer = Input(shape=(sequences, features), name='piano_noise_mixed')
        # input_layer = Input(shape=(sequences, features), dtype='float16', 
        #                     name='piano_noise_mixed')

        if config is not None:
            num_layers = len(config['layers'])
            prev_layer_type = None  # Works b/c all RNN stacks are size > 1
            for i in range(num_layers):
                layer_config = config['layers'][i]
                curr_layer_type = layer_config['type']

                # Standardize option
                if config['scale'] and i == 0:
                    x = Standardize(t_mean, t_std) (input_layer)

                # Add skip connection if necessary
                if (config['rnn_res_cntn'] and prev_layer_type is not None and
                    prev_layer_type != 'Dense' and curr_layer_type == 'Dense'):
                    x = Concatenate() ([x, input_layer])
        
                if curr_layer_type == 'RNN':
                    if config['bidir']:
                        x = Bidirectional(SimpleRNN(features // layer_config['nrn_div'], 
                                activation=layer_config['act'], 
                                use_bias=config['bias_rnn'],
                                dropout=config['rnn_dropout'][0],
                                recurrent_dropout=config['rnn_dropout'][1],
                                return_sequences=True)) (input_layer if (i == 0 and not config['scale']) else x)
                    else:
                        x = SimpleRNN(features // layer_config['nrn_div'], 
                                activation=layer_config['act'], 
                                use_bias=config['bias_rnn'],
                                dropout=config['rnn_dropout'][0],
                                recurrent_dropout=config['rnn_dropout'][1],
                                return_sequences=True) (input_layer if (i == 0 and not config['scale']) else x)

                elif curr_layer_type == 'LSTM':
                    if config['bidir']:
                        x = Bidirectional(LSTM(features // layer_config['nrn_div'], 
                                activation=layer_config['act'], 
                                use_bias=config['bias_rnn'],
                                dropout=config['rnn_dropout'][0],
                                recurrent_dropout=config['rnn_dropout'][1],
                                return_sequences=True)) (input_layer if (i == 0 and not config['scale']) else x)
                    else:
                        x = LSTM(features // layer_config['nrn_div'], 
                                activation=layer_config['act'], 
                                use_bias=config['bias_rnn'],
                                dropout=config['rnn_dropout'][0],
                                recurrent_dropout=config['rnn_dropout'][1],
                                return_sequences=True) (input_layer if (i == 0 and not config['scale']) else x)
                elif curr_layer_type == 'Dense':
                    if i == (num_layers - 1):   # Last layer is fork layer
                        # Reverse standardization at end of model if appropriate
                        if config['scale']:
                            piano_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                            activation=layer_config['act'], 
                                                            use_bias=config['bias_dense']), 
                                                        name='piano_hat'
                                                    ) (x)
                            noise_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                            activation=layer_config['act'], 
                                                            use_bias=config['bias_dense']), 
                                                        name='noise_hat'
                                                    ) (x)
                            if config['bn']:
                                piano_hat = BatchNormalization() (piano_hat)
                                noise_hat = BatchNormalization() (noise_hat)
                            
                            piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
                            noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
                    
                        else:
                            piano_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                            activation=layer_config['act'], 
                                                            use_bias=config['bias_dense']), 
                                                        name='piano_hat'
                                                    ) (x)
                            noise_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                            activation=layer_config['act'], 
                                                            use_bias=config['bias_dense']),
                                                        name='noise_hat'
                                                    ) (x)
                            if config['bn']:
                                piano_hat = BatchNormalization() (piano_hat)
                                noise_hat = BatchNormalization() (noise_hat)

                    else:
                        x = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                activation=layer_config['act'], 
                                                use_bias=config['bias_dense']), 
                                        ) (input_layer if (i == 0 and not config['scale']) else x)
                        if config['bn']:
                            x = BatchNormalization() (x)

                prev_layer_type = curr_layer_type
        
        elif test > 0:
            if test == 1:
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            # The difference in mem use between test 1 @ 2 is much an long a rnn takes
            if test == 2:
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            if test == 3:
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            if test == 4:
                x = LSTM(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = LSTM(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x)
                x = LSTM(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            # The difference in mem use between test 2 @ 4 is how much more an lstm takes
            if test == 5:
                x = LSTM(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = LSTM(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            if test == 6:
                x = LSTM(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer)
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            if test == 7:
                x = Standardize(t_mean, t_std) (input_layer)
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
                noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
            # The difference in mem use between test 6 & 2 is what doubling dim red does to mem usage
            if test == 8:
                x = Standardize(t_mean, t_std) (input_layer)
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
                noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
            if test == 9:
                x = Standardize(t_mean, t_std) (input_layer)
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
                noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
            if test == 10:
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x)
                x = Concatenate() ([x, input_layer])
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            # The difference in mem use between test 1 @ 2 is much an long a rnn takes
            if test == 11:
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (x) 
                x = Concatenate() ([x, input_layer])
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            if test == 12:
                x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = Concatenate() ([x, input_layer])
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

            if test == 13:
                x = Bidirectional(SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True)) (input_layer) 
                x = Bidirectional(SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True)) (x) 
                x = Bidirectional(SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True)) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            # The difference in mem use between test 1 @ 2 is much an long a rnn takes
            if test == 14:
                x = Bidirectional(SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True)) (input_layer) 
                x = Bidirectional(SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True)) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            if test == 15:
                x = Bidirectional(SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True)) (input_layer) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

            if test == 16:
                x = SimpleRNN(features - 1, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = SimpleRNN(features - 1, 
                        activation='relu', 
                        return_sequences=True) (x) 
                x = SimpleRNN(features - 1, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            # The difference in mem use between test 6 & 2 is what doubling dim red does to mem usage
            if test == 17:
                x = SimpleRNN(features - 1, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                x = SimpleRNN(features - 1, 
                        activation='relu', 
                        return_sequences=True) (x) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            if test == 18:
                x = SimpleRNN(features - 1, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
                piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
                noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

        # Use pre-configurations (default)
        else:
            x = SimpleRNN(features // 2, 
                        activation='relu', 
                        return_sequences=True) (input_layer) 
            x = SimpleRNN(features // 2, 
                    activation='relu',
                    return_sequences=True) (x)
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        piano_pred = TimeFreqMasking(epsilon=epsilon, 
                                    name='piano_pred') ((piano_hat, noise_hat, input_layer))
        noise_pred = TimeFreqMasking(epsilon=epsilon, 
                                    name='noise_pred') ((noise_hat, piano_hat, input_layer))

        # model = Model(inputs=input_layer, outputs=[piano_pred, noise_pred])

        # # Keras debug block
        # debug_piano_model = Model(
        #     inputs=model.inputs,
        #     # inputs=model.layers[3].output,
        #     # outputs=[model.layers[0].output] + model.outputs,
        #     outputs=[model.layers[2].output, model.layers[3].output, model.layers[5].output],
        #     name='Debug Piano Model (rnn2 out -> piano_hat out -> piano_pred out)'
        # )
        # debug_noise_model = Model(
        #     inputs=model.inputs,
        #     outputs=[model.layers[2].output, model.layers[4].output, model.layers[6].output],
        #     name='Debug Noise Model (rnn2 out -> noise_hat out -> noise_pred out)'
        # )
        # xs = tf.random.normal((3, sequences, features))
        # # print('DEBUG Piano Model Summary:')
        # # print(debug_piano_model.summary())
        # print('DEBUG Piano Model Run:')
        # # print(debug_piano_model(xs, training=True))

        # debug_piano_model_outputs = debug_piano_model(xs, training=True)
        # rnn_o, dense_o, mask_o = debug_piano_model_outputs[0].numpy(), debug_piano_model_outputs[1].numpy(), debug_piano_model_outputs[2].numpy()
        # print('Shape rnn out:', rnn_o.shape)
        # print('Shape dense out:', dense_o.shape)
        # print('Shape mask out:', mask_o.shape)
        # # print('Inf in rnn out:', True in np.isinf(rnn_o))
        # # print('Inf in dense out:', True in np.isinf(dense_o))
        # # print('Inf in mask out:', True in np.isinf(mask_o))
        # # print('NaN in rnn out:', True in np.isnan(rnn_o))
        # # print('NaN in dense out:', True in np.isnan(dense_o))
        # # print('NaN in mask out:', True in np.isnan(mask_o))
        # print()

        # # print('DEBUG Noise Model Summary:')
        # # print(debug_noise_model.summary())
        # print('DEBUG Noise Model Run:')
        # # print(debug_noise_model(xs, training=True))
        # debug_noise_model_outputs = debug_noise_model(xs, training=True)
        # rnn_o, dense_o, mask_o = debug_noise_model_outputs[0].numpy(), debug_noise_model_outputs[1].numpy(), debug_noise_model_outputs[2].numpy()
        # print('Shape rnn out:', rnn_o.shape)
        # print('Shape dense out:', dense_o.shape)
        # print('Shape mask out:', mask_o.shape)
        # # print('Inf in rnn out:', True in np.isinf(rnn_o))
        # # print('Inf in dense out:', True in np.isinf(dense_o))
        # # print('Inf in mask out:', True in np.isinf(mask_o))
        # # print('NaN in rnn out:', True in np.isnan(rnn_o))
        # # print('NaN in dense out:', True in np.isnan(dense_o))
        # # print('NaN in mask out:', True in np.isnan(mask_o))
        # print()
        # # print('Model Layers:')
        # # print([layer.name for layer in model.layers])
        # # ['piano_noise_mixed', 'simple_rnn', 'simple_rnn_1', 'piano_hat', 'noise_hat', 'piano_pred', 'noise_pred']    
        
        # disc_loss = None
        if pre_trained_wgts is not None:
                # loss_const_tensor = tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
                preds_and_lc = Concatenate(axis=0) ([piano_pred, 
                                                    noise_pred, 
                                                    #  loss_const_tensor
                                                    tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
                                                    ])
                # MIXED PRECISION - not sure if test case is necessary
                # if not pc_run:
                #     preds_and_lc = Activation('linear', name='mp_output', dtype='float32') (preds_and_lc)

                model = Model(inputs=input_layer, outputs=preds_and_lc)

                print('Only loading pre-trained weights for prediction')
                model.set_weights(pre_trained_wgts)
        elif keras_fit:
            # # print('MODEL OUTPUT TYPES:', piano_pred.dtype, noise_pred.dtype)
            # # print('MODEL TARGETS:', piano_pred.dtype, noise_pred.dtype)
            # # TEST FLOAT16
            # piano_true = Input(shape=(sequences, features), name='piano_true')
            # noise_true = Input(shape=(sequences, features), name='noise_true')
            # # piano_true = Input(shape=(sequences, features), dtype='float32', 
            # #                 name='piano_true')
            # # noise_true = Input(shape=(sequences, features), dtype='float32', 
            # #                 name='noise_true')
            # model = Model(inputs=[input_layer, piano_true, noise_true],
            #         outputs=[piano_pred, noise_pred])

            # loss_const = tf.constant(loss_const) # For performance/less mem
            # # FLOAT16 TEST
            # # loss_const = tf.dtypes.cast(loss_const, tf.float16)
            # # print('MODEL LOSS_CONST:', loss_const.dtype)
            # # 1 val instead of 1 val/batch makes for less mem used
            # last_dim = noise_pred.shape[1] * noise_pred.shape[2]
            # disc_loss = (
            #     tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2) - 
            #     (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2)) +
            #     tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2) -
            #     (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2))
            # )
            # # # FLOAT16 TEST
            # # disc_loss = (
            # #     tf.dtypes.cast(tf.math.reduce_mean(tf.dtypes.cast(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)), tf.float16) ** 2, axis=-1), tf.float16) - 
            # #     (loss_const * tf.dtypes.cast(tf.math.reduce_mean(tf.dtypes.cast(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)), tf.float16) ** 2, axis=-1), tf.float16)) +
            # #     tf.dtypes.cast(tf.math.reduce_mean(tf.dtypes.cast(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)), tf.float16) ** 2, axis=-1), tf.float16) -
            # #     (loss_const * tf.dtypes.cast(tf.math.reduce_mean(tf.dtypes.cast(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)), tf.float16) ** 2, axis=-1), tf.float16))
            # # )
            # # OOM BUG TEST - change one factor - loss
            # # model.add_loss(disc_loss)
            # # model.compile(optimizer=optimizer, loss={'piano_pred': 'mse', 'noise_pred': 'mse'})

            # loss_const_tensor = tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
            preds_and_lc = Concatenate(axis=0) ([piano_pred, 
                                                noise_pred, 
                                                #  loss_const_tensor
                                                tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
                                                ])
            # MIXED PRECISION
            # if not pc_run:
            #     preds_and_lc = Activation('linear', name='mp_output', dtype='float32') (preds_and_lc)

            model = Model(inputs=input_layer, outputs=preds_and_lc)
            # model = Model(inputs=input_layer, outputs=output)

            # Combine piano_pred, noise_pred & loss_const into models output!
            # loss_const_tensor = tf.reshape(tf.constant(loss_const), [None, sequences, 1])
            # output = Concatenate() ([piano_pred, noise_pred, loss_const_tensor])

            model.compile(optimizer=optimizer, loss=discrim_loss)
            # TRY - B/C tf.function is symbolic tensors - no error there
            # Assign loss to each output -> keras should average/sum it
            # # Problematic, returns a func or eager func, not a tensor
            # @tf.function
            # def piano_loss(noise_true, noise_pred, loss_const):
            #     def closure(piano_true, piano_pred):
            #         last_dim = noise_pred.shape[1] * noise_pred.shape[2]
            #         return (
            #             tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
            #             (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
            #             tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) -
            #             (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1))
            #         )
            #     return closure

            # @tf.function
            # def noise_loss(piano_true, piano_pred, loss_const):
            #     def closure(noise_true, noise_pred):
            #         last_dim = piano_pred.shape[1] * piano_pred.shape[2]
            #         return (
            #             tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
            #             (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
            #             tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) -
            #             (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1))
            #         )
            #     return closure

            # model.compile(optimizer=optimizer,
            #               loss={
            #                     'piano_pred': custom_loss(piano_true, piano_pred, noise_true, noise_pred, loss_const),
            #                     'noise_pred': custom_loss(noise_true, noise_pred, piano_true, piano_pred, loss_const)
            #                    })
            #                 #   loss={
            #                 #       'piano_pred': piano_loss(noise_true, noise_pred, loss_const),
            #                 #       'noise_pred': noise_loss(piano_true, piano_pred, loss_const)
            #                 #        })

        return model    #, disc_loss

    def make_gen_callable(_gen):
        def gen():
            for x,y in _gen:
                yield x,y
        return gen

    train_dataset = tf.data.Dataset.from_generator(
        make_gen_callable(train_generator), 
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, n_seq, n_feat), (None, n_seq, n_feat*2)),
    )
    val_dataset = tf.data.Dataset.from_generator(
        make_gen_callable(validation_generator), 
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, n_seq, n_feat), (None, n_seq, n_feat*2)),
    )


    train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

    # train_dataset = np.random.rand(num_train, n_seq, n_feat)

    print('Making model...')
    # if pc_run:
    model = make_model(n_feat, n_seq, name='Training Model', epsilon=epsilon, loss_const=loss_const,
                            config=config, t_mean=t_mean, t_std=t_std, optimizer=optimizer,
                            pc_run=pc_run, keras_fit=keras_fit)
    # print('KERAS LOSS TENSOR:', keras_fit_loss)
    
        # optimizer = optimizer
    # else:
    #     with mirrored_strategy.scope():
    #         model = make_model(n_feat, n_seq, name='Training Model', epsilon=epsilon, 
    #                                 config=config, t_mean=t_mean, t_std=t_std)
    #         optimizer = optimizer
    print(model.summary())

    # MIXED PRECISION
    # if not pc_run:
        # optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = '../logs/gradient_tape/' + current_time + '/train'
    # test_log_dir = '../logs/gradient_tape/' + current_time + '/test'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    print('Going into training now...')
    hist = model.fit(train_dataset,
                    steps_per_epoch=math.ceil(num_train / batch_size),
                    epochs=epochs,
                    validation_data=val_dataset,
                    validation_steps=math.ceil(num_val / batch_size),
                    callbacks=[EarlyStopping('val_loss', patience=patience, mode='min')])#,
                            # Done memory profiling
                            # TensorBoard(log_dir=log_dir, profile_batch='2, 4')])   # 10' # by default, profiles 2nd batch
    history = hist.history
    
    # import tensorflow as tf
    # # from tensorflow import keras
    # # from tensorflow.keras.losses import Loss
    # from tensorflow.keras.layers import Input, SimpleRNN, Dense, Lambda, TimeDistributed, Layer, LSTM, Bidirectional, BatchNormalization, Concatenate, Activation
    # from tensorflow.keras.models import Model
    # # from tensorflow.keras.utils import Sequence
    # # from tensorflow.keras.activations import relu
    # from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

    # physical_gpus = tf.config.list_physical_devices('GPU')
    # # if (not pc_run) and physical_gpus:
    # # # No courtesy, allocate all GPU mem for me only, guarantees no else can mess up me. If no avail GPUs, admin's fault
    # # # mostly courtesy to others on F35 system
    # # print("Setting memory growth on GPUs")
    # # for i in range(len(physical_gpus)):
    # #     tf.config.experimental.set_memory_growth(physical_gpus[i], True)

    # # Restrict TensorFlow to only use one GPU (exclusive access w/ mem growth = False), F35 courtesy
    # try:
    #     # Pick random GPU, 1/10 times, pick GPU 0 other times 
    #     # (pick same GPU most times, but be prepared for it to be taken between restarts)
    #     choice = random.randint(0, 9)
    #     chosen_gpu = random.randint(0, len(physical_gpus)-1) if (choice == 0) else 0
    #     print("Restricting TF run to only use 1 GPU:", physical_gpus[chosen_gpu], "\n")
    #     tf.config.experimental.set_visible_devices(physical_gpus[chosen_gpu], 'GPU')
    #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #     print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    # except RuntimeError as e:
    #     # Visible devices must be set before GPUs have been initialized
    #     print(e)

    # time.sleep(1)
    # list1, list2 = [1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]
    print('PID:', os.getpid())
    send_end.send((history['loss'], history['val_loss']))

def main():
    optimizer = tf.keras.optimizers.RMSprop()
    num_train, num_val, n_feat, n_seq, batch_size, loss_const, epochs = 45, 15, 2049, 1000, 15, 0.05, 1

    pc_run = False

    t_gen = fixed_data_generator(num_train,
                                    batch_size=batch_size, num_seq=n_seq, num_feat=n_feat, pc_run=pc_run)
    v_gen = fixed_data_generator(num_val,
                    batch_size=batch_size, num_seq=n_seq, num_feat=n_feat, pc_run=pc_run)

    for i in range(10000):
        # ctx = mp.get_context('spawn')
        send_end, recv_end = multiprocessing.Pipe()
        process_train = multiprocessing.Process(target=test_func, args=(i, send_end,
            t_gen, v_gen,
            num_train, num_val, n_feat, n_seq, batch_size, 
                        loss_const, epochs, #epochs=20, 
                        optimizer, # optimizer=None,
                        # # optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.75),
                        # patience=100, epsilon=10 ** (-10), config=None, recent_model_path=None, pc_run=False,
                        # t_mean=None, t_std=None, grid_search_iter=None, gs_path=None, combos=None, gs_id='',
                        # keras_fit=False, ret_queue=None
        ))

        process_train.start()
        losses, val_losses = recv_end.recv()                    
        process_train.join()

        print('Iter', i, 'losses:', losses)
        print('Iter', i, 'val losses:', val_losses)

if __name__ == '__main__':
    main()