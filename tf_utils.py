import tensorflow as tf


def temporal_convolution_layer(inputs, output_units, convolution_width, causal=False, dilation_rate=[1], bias=True,
                               activation=None, dropout=None, scope='temporal-convolution-layer', reuse=False):
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
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        z = z[:, :-shift, :] if causal else z
        return z


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

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]


def sequence_smape(y, y_hat, sequence_lengths, is_nan):
    max_sequence_length = tf.shape(y)[1]
    y = tf.cast(y, tf.float32)
    smape = 2*(tf.abs(y_hat - y) / (tf.abs(y) + tf.abs(y_hat)))

    # ignore discontinuity
    zero_loss = 2.0*tf.ones_like(smape)
    nonzero_loss = smape
    smape = tf.where(tf.logical_or(tf.equal(y, 0.0), tf.equal(y_hat, 0.0)), zero_loss, nonzero_loss)

    sequence_mask = tf.cast(tf.sequence_mask(sequence_lengths, maxlen=max_sequence_length), tf.float32)
    sequence_mask = sequence_mask*(1 - is_nan)
    avg_smape = tf.reduce_sum(smape*sequence_mask) / tf.reduce_sum(sequence_mask)
    return avg_smape


def sequence_mean(x, lengths):
    return tf.reduce_sum(x, axis=1) / tf.cast(lengths, tf.float32)
