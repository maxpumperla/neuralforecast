# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.layers.recurrent import SimpleRNN


class ARMA(SimpleRNN):
    '''
    Recurrent neural network layer derived from a fully-connected RNN, replicating the structure
    of an ARMA model used for time-series prediction.
    Time-series of input shape (batch_size, time, input_dim) feed into this layer and produce an
    output of shape (batch_size, output_dim). output_dim has to be one for this layer to work.
    Internally, a state vector of length q is maintained, which represents a length q memory cell
    of hidden values. The output (length one) of this layer at time t-1 will be appended to the
    state vector for time t and the previously last state will be dropped.

    Parameters:
        q: Input dimension of recurrent/context units, may differ from output dim
    '''
    def __init__(self, q, ma_only=False, **kwargs):
        self.inner_input_dim = q
        self.ma_only = ma_only
        super(ARMA, self).__init__(**kwargs)

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, inner_input_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.inner_input_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, inner_input_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def preprocess_input(self, x):
            return x

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        # Only change in build compared to SimpleRNN:
        # U is of shape (inner_input_dim, output_dim) now.
        self.U = self.inner_init((self.inner_input_dim, self.output_dim),
                                 name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        hidden_input = states[0]
        B_U = states[1]
        B_W = states[2]

        # Make last hidden input the residual of the prediction and
        # the last available feature.
        if self.inner_input_dim > 0:
            update = K.expand_dims(hidden_input[:, -1] - x[:, -1])
            hidden_input = K.concatenate((hidden_input[:, :-1], update))

        if self.ma_only:
            h = self.b
        else:
            h = K.dot(x * B_W, self.W) + self.b

        if self.inner_input_dim > 0:
            output = self.activation(h + K.dot(hidden_input * B_U, self.U))
            new_state = K.concatenate((hidden_input[:, 1:], output))
            return output, [new_state]

        else:
            output = self.activation(h)
            return output, [output]

    def get_config(self):
        config = {"inner_input_dim": self.inner_input_dim}
        base_config = super(ARMA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InputSpec(object):
    '''
    Note: This has been copied from Keras' engine module, which is not importable.

    This specifies the ndim, dtype and shape of every input to a layer.
    Every layer should expose (if appropriate) an `input_spec` attribute:
    a list of instances of InputSpec (one per input tensor).
    A None entry in a shape is compatible with any dimension,
    a None shape is compatible with any shape.
    '''
    def __init__(self, dtype=None, shape=None, ndim=None):
        if type(ndim) is str:
            assert '+' in ndim, 'When passing a str "ndim", it should have the form "2+", "3+", etc.'
            int_ndim = ndim[:ndim.find('+')]
            assert int_ndim.isdigit(), 'When passing a str "ndim", it should have the form "2+", "3+", etc.'
        if shape is not None:
            self.ndim = len(shape)
        else:
            self.ndim = ndim
        self.dtype = dtype
        self.shape = shape
