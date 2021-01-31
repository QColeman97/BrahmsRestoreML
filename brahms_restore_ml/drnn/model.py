import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, Dense, TimeDistributed, Layer, LSTM, Bidirectional, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

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
    def __init__(self, epsilon, **kwargs):
        super(TimeFreqMasking, self).__init__(**kwargs)
        self.epsilon = epsilon

    # No build method, b/c passing in multiple inputs to layer (no single shape)

    def call(self, inputs):
        y_hat_self, y_hat_other, x_mixed = inputs
        mask = tf.abs(y_hat_self) / (tf.abs(y_hat_self) + tf.abs(y_hat_other) + self.epsilon)
        y_tilde_self = mask * x_mixed
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

    # True sum of L2 Norm doesn't divide by N (want that)
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
                    pc_run=False):
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
    
    # Combine piano_pred, noise_pred & loss_const into models output!
    if pre_trained_wgts is not None:
        preds_and_gamma = Concatenate(axis=0) ([piano_pred, 
                                            noise_pred, 
                                            #  loss_const_tensor
                                            tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
                                            ])
        model = Model(inputs=input_layer, outputs=preds_and_gamma)

        print('Only loading pre-trained weights for prediction')
        model.set_weights(pre_trained_wgts)
    else:
        preds_and_gamma = Concatenate(axis=0) ([piano_pred, 
                                            noise_pred, 
                                            #  loss_const_tensor
                                            tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
                                            ])
        model = Model(inputs=input_layer, outputs=preds_and_gamma)
        model.compile(optimizer=optimizer, loss=discrim_loss)

    return model
