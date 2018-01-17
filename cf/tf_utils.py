import tensorflow as tf


def temporal_convolution_layer(inputs, output_units, convolution_width, causal=False, dilation_rate=[1], bias=True,
                               activation=None, dropout=None, scope='temporal-convolution-layer', reuse=False,
                               batch_norm=None):
    """
    Convolution over the temporal axis of sequence data.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of timesteps to use in convolution.
        causal: Output at timestep t is a function of inputs at or before timestep t.
        dilation_rate:  Dilation rate along temporal axis.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        if causal:
            shift = (convolution_width / 2) + (int(dilation_rate[0] - 1) / 2)
            pad = tf.zeros([tf.shape(inputs)[0], shift, inputs.shape.as_list()[2]])
            inputs = tf.concat([pad, inputs], axis=1)

        W = tf.get_variable(
            name='weights',
            initializer=tf.random_normal_initializer(
                mean=0,
                stddev=1.0 / tf.sqrt(float(convolution_width)*float(shape(inputs, 2)))
            ),
            shape=[convolution_width, shape(inputs, 2), output_units]
        )

        z = tf.nn.convolution(inputs, W, padding='SAME', dilation_rate=dilation_rate)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        z = tf.layers.batch_normalization(z, training=batch_norm) if batch_norm is not None else z
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        z = z[:, :-shift, :] if causal else z
        return z


def bidirectional_lstm_layer(inputs, lengths, state_size, scope='bi-lstm-layer', reuse=False):
    """
    Bidirectional LSTM layer.
    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.
    Returns:
        Tensor of shape [batch size, max sequence length, 2*state_size] containing the concatenated
        forward and backward lstm outputs at each timestep.
    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.LSTMCell(
            state_size,
            reuse=reuse
        )
        cell_bw = tf.contrib.rnn.LSTMCell(
            state_size,
            reuse=reuse
        )
        outputs, (output_fw, output_bw) = tf.nn.bidirectional_dynamic_rnn(
            inputs=inputs,
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        outputs = tf.concat(outputs, 2)
        return outputs


def time_distributed_dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                                 dropout=None, scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0 / float(shape(inputs, -1))),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        z = tf.layers.batch_normalization(z, training=batch_norm) if batch_norm is not None else z
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def sequence_rmse(y, y_hat, sequence_lengths, weights):
    """
    Calculate weighted RMSE(y, y_hat).

    Args:
        y: Label tensor of shape [batch_size, timesteps]
        y_hat: Prediction tensor of shape [batch_size, timesteps]
        sequence_lengths: Length of sequences, tensor of shape [batch_size]
        weights: Weights for each sequence, tensor of shape [batch_size]

    Returns:
        RMSE as a 0-dimensional tensor

    """
    square_error = tf.square(y_hat - y)
    sequence_mask = tf.cast(tf.sequence_mask(sequence_lengths, maxlen=tf.shape(y)[1]), tf.float32)
    weights = sequence_mask*tf.expand_dims(weights, 1)
    avg_square_error = tf.reduce_sum(square_error*weights) / tf.reduce_sum(weights)
    return tf.sqrt(avg_square_error)


def sequence_mean(x, lengths):
    """
    Compute mean across temporal axis of sequence.

    Args:
        x: Tensor of shape [batch_size, timesteps]
        lengths: Lengths of sequences, tensor of shape [batch_size]

    Returns:
        Sequence means as tensor of shape [batch_size]
    """
    sequence_mask = tf.cast(tf.sequence_mask(lengths, maxlen=tf.shape(x)[1]), tf.float32)
    mean = tf.reduce_sum(x*sequence_mask, axis=1) / (tf.cast(lengths, tf.float32))
    return tf.where(tf.is_nan(mean), tf.zeros_like(mean), mean)


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]
