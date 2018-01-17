import os

import numpy as np
import tensorflow as tf

from data_frame import DataFrame
from tf_base_model import TFBaseModel
from tf_utils import (
    time_distributed_dense_layer, temporal_convolution_layer,
    sequence_mean, sequence_rmse, shape, bidirectional_lstm_layer
)


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'x_raw',
            'onpromotion',
            'id',
            'x',
            'store_nbr',
            'item_nbr',
            'city',
            'state',
            'type',
            'cluster',
            'family',
            'class',
            'perishable',
            'is_discrete',
            'start_date',
            'x_lags',
            'xy_lags',
            'ts',
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95)

        self.num_city = self.test_df['city'].max() + 1
        self.num_state = self.test_df['state'].max() + 1
        self.num_type = self.test_df['type'].max() + 1
        self.num_cluster = self.test_df['cluster'].max() + 1
        self.num_family = self.test_df['family'].max() + 1
        self.num_item_class = self.test_df['class'].max() + 1
        self.num_perishable = self.test_df['perishable'].max() + 1
        self.num_store_nbr = self.test_df['store_nbr'].max() + 1
        self.num_item_nbr = self.test_df['item_nbr'].max() + 1

        print 'train size', len(self.train_df)
        print 'val size', len(self.val_df)
        print 'test size', len(self.test_df)

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            mode='train'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            mode='val'
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=1,
            mode='test'
        )

    def batch_generator(self, batch_size, df, mode, shuffle=True, num_epochs=10000):
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode == 'test')
        )
        for batch in batch_gen:
            num_decode_steps = 16
            full_seq_len = batch['x'].shape[1] - num_decode_steps
            max_encode_length = full_seq_len

            x = np.zeros([len(batch), max_encode_length])
            y = np.zeros([len(batch), num_decode_steps])
            x_raw = np.zeros([len(batch), max_encode_length])
            x_lags = np.zeros([len(batch), max_encode_length, batch['x_lags'].shape[2] + batch['xy_lags'].shape[2]])
            y_lags = np.zeros([len(batch), num_decode_steps, batch['xy_lags'].shape[2]])
            x_op = np.zeros([len(batch), max_encode_length])
            y_op = np.zeros([len(batch), num_decode_steps])
            x_len = np.zeros([len(batch)])
            y_len = np.zeros([len(batch)])
            x_idx = np.zeros([len(batch), max_encode_length])
            y_idx = np.zeros([len(batch), num_decode_steps])
            y_id = np.zeros([len(batch), num_decode_steps])
            x_ts = np.zeros([len(batch), max_encode_length, batch['ts'].shape[2]])
            weights = np.zeros([len(batch)])
            weights[batch['perishable'] == 1] = 1.25
            weights[batch['perishable'] == 0] = 1.0

            for i, (data, data_raw, start_idx, x_lag, xy_lag, op, uid, ts) in enumerate(zip(
                    batch['x'], batch['x_raw'], batch['start_date'], batch['x_lags'],
                    batch['xy_lags'], batch['onpromotion'], batch['id'], batch['ts']
                )
            ):
                seq_len = full_seq_len - start_idx
                val_window = 365
                train_window = 365

                if mode == 'train':
                    if seq_len == 0:
                        rand_encode_len = 0
                        weights[i] = 0
                    elif seq_len <= train_window:
                        rand_encode_len = np.random.randint(0, seq_len)
                    else:
                        rand_encode_len = np.random.randint(seq_len - train_window, seq_len)
                    rand_decode_len = min(seq_len - rand_encode_len, num_decode_steps)

                elif mode == 'val':
                    if seq_len <= num_decode_steps:
                        rand_encode_len = 0
                        weights[i] = 0
                    elif seq_len <= val_window + num_decode_steps:
                        rand_encode_len = np.random.randint(0, seq_len - num_decode_steps + 1)
                    else:
                        rand_encode_len = np.random.randint(
                            seq_len - (val_window + num_decode_steps), seq_len - num_decode_steps + 1)
                    rand_decode_len = min(seq_len - rand_encode_len, num_decode_steps)

                elif mode == 'test':
                    rand_encode_len = seq_len
                    rand_decode_len = num_decode_steps

                end_idx = start_idx + rand_encode_len

                x[i, :rand_encode_len] = data[start_idx: end_idx]
                y[i, :rand_decode_len] = data[end_idx: end_idx + rand_decode_len]
                x_raw[i, :rand_encode_len] = data_raw[start_idx: end_idx]

                x_lags[i, :rand_encode_len, :x_lag.shape[1]] = x_lag[start_idx: end_idx, :]
                x_lags[i, :rand_encode_len, x_lag.shape[1]:] = xy_lag[start_idx: end_idx, :]
                y_lags[i, :rand_decode_len, :] = xy_lag[end_idx: end_idx + rand_decode_len, :]

                x_op[i, :rand_encode_len] = op[start_idx: end_idx]
                y_op[i, :rand_decode_len] = op[end_idx: end_idx + rand_decode_len]
                x_ts[i, :rand_encode_len, :] = ts[start_idx: end_idx, :]
                x_idx[i, :rand_encode_len] = np.floor(np.log(np.arange(rand_encode_len) + 1))
                y_idx[i, :rand_decode_len] = np.floor(
                    np.log(np.arange(rand_encode_len, rand_encode_len + rand_decode_len) + 1))
                y_id[i, :rand_decode_len] = uid[end_idx: end_idx + rand_decode_len]
                x_len[i] = end_idx - start_idx
                y_len[i] = rand_decode_len

            batch['x_'] = batch['x']
            batch['x'] = x
            batch['y'] = y
            batch['x_raw'] = x_raw
            batch['x_lags'] = x_lags
            batch['y_lags'] = y_lags
            batch['x_op'] = x_op
            batch['y_op'] = y_op
            batch['x_ts'] = x_ts
            batch['x_idx'] = x_idx
            batch['y_idx'] = y_idx
            batch['y_id'] = y_id
            batch['x_len'] = x_len
            batch['y_len'] = y_len
            batch['item_class'] = batch['class']
            batch['weights'] = weights

            yield batch


class cnn(TFBaseModel):

    def __init__(
        self,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(8)]*3,
        filter_widths=[2 for i in range(8)]*3,
        num_decode_steps=16,
        **kwargs
    ):
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.filter_widths = filter_widths
        self.num_decode_steps = num_decode_steps
        super(cnn, self).__init__(**kwargs)

    def get_input_sequences(self):
        self.x = tf.placeholder(tf.float32, [None, None])
        self.y = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.x_raw = tf.placeholder(tf.float32, [None, None])
        self.x_len = tf.placeholder(tf.int32, [None])
        self.y_len = tf.placeholder(tf.int32, [None])
        self.x_op = tf.placeholder(tf.int32, [None, None])
        self.y_op = tf.placeholder(tf.int32, [None, self.num_decode_steps])
        self.x_id = tf.placeholder(tf.int32, [None, None])
        self.y_id = tf.placeholder(tf.int32, [None, self.num_decode_steps])
        self.x_idx = tf.placeholder(tf.int32, [None, None])
        self.y_idx = tf.placeholder(tf.int32, [None, self.num_decode_steps])

        self.x_dow = tf.placeholder(tf.int32, [None, None])
        self.y_dow = tf.placeholder(tf.int32, [None, self.num_decode_steps])
        self.x_month = tf.placeholder(tf.int32, [None, None])
        self.y_month = tf.placeholder(tf.int32, [None, self.num_decode_steps])
        self.x_week = tf.placeholder(tf.int32, [None, None])
        self.y_week = tf.placeholder(tf.int32, [None, self.num_decode_steps])

        self.x_ts = tf.placeholder(tf.float32, [None, None, 16])
        self.x_lags = tf.placeholder(tf.float32, [None, None, 12])
        self.y_lags = tf.placeholder(tf.float32, [None, self.num_decode_steps, 9])

        self.city = tf.placeholder(tf.int32, [None])
        self.state = tf.placeholder(tf.int32, [None])
        self.type = tf.placeholder(tf.int32, [None])
        self.cluster = tf.placeholder(tf.int32, [None])
        self.family = tf.placeholder(tf.int32, [None])
        self.item_class = tf.placeholder(tf.int32, [None])
        self.perishable = tf.placeholder(tf.int32, [None])
        self.is_discrete = tf.placeholder(tf.float32, [None])
        self.store_nbr = tf.placeholder(tf.int32, [None])
        self.item_nbr = tf.placeholder(tf.int32, [None])

        self.weights = tf.placeholder(tf.float32, [None])

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        item_class_embeddings = tf.get_variable(
            name='item_class_embeddings',
            shape=[self.reader.num_item_class, 20],
            dtype=tf.float32
        )
        item_class = tf.nn.embedding_lookup(item_class_embeddings, self.item_class)

        item_nbr_embeddings = tf.get_variable(
            name='item_nbr_embeddings',
            shape=[self.reader.num_item_nbr, 50],
            dtype=tf.float32
        )
        item_nbr = tf.nn.embedding_lookup(item_nbr_embeddings, self.item_nbr)

        self.x_mean = tf.expand_dims(sequence_mean(self.x, self.x_len), 1)
        self.x_centered = self.x - self.x_mean
        self.y_centered = self.y - self.x_mean
        self.x_ts_centered = self.x_ts - tf.expand_dims(self.x_mean, 2)
        self.x_lags_centered = self.x_lags - tf.expand_dims(self.x_mean, 2)
        self.y_lags_centered = self.y_lags - tf.expand_dims(self.x_mean, 2)
        self.x_is_zero = tf.cast(tf.equal(self.x_raw, tf.zeros_like(self.x_raw)), tf.float32)
        self.x_is_negative = tf.cast(tf.less(self.x_raw, tf.zeros_like(self.x_raw)), tf.float32)

        self.encode_features = tf.concat([
            self.x_ts_centered,
            self.x_lags_centered,
            tf.one_hot(self.x_op, 3),
            tf.one_hot(self.x_idx, 9),
            tf.expand_dims(self.x_is_zero, 2),
            tf.expand_dims(self.x_is_negative, 2),
            tf.tile(tf.expand_dims(self.x_mean, 2), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.city, self.reader.num_city), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.state, self.reader.num_state), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.type, self.reader.num_type), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.cluster, self.reader.num_cluster), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.family, self.reader.num_family), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(item_class, 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.perishable, self.reader.num_perishable), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.expand_dims(self.is_discrete, 1), 2), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.store_nbr, self.reader.num_store_nbr), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(item_nbr, 1), (1, tf.shape(self.x)[1], 1)),
        ], axis=2)

        decode_idx = tf.tile(tf.expand_dims(tf.range(self.num_decode_steps), 0), (tf.shape(self.y)[0], 1))
        decode_features = tf.concat([
            self.y_lags_centered,
            tf.one_hot(decode_idx, self.num_decode_steps),
            tf.one_hot(self.y_op, 3),
            tf.one_hot(self.y_idx, 9),
            tf.tile(tf.expand_dims(self.x_mean, 2), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.city, self.reader.num_city), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.state, self.reader.num_state), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.type, self.reader.num_type), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.cluster, self.reader.num_cluster), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.family, self.reader.num_family), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(item_class, 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.perishable, self.reader.num_perishable), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.expand_dims(self.is_discrete, 1), 2), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.store_nbr, self.reader.num_store_nbr), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(item_nbr, 1), (1, self.num_decode_steps, 1)),
        ], axis=2)

        lstm_decode_features = bidirectional_lstm_layer(decode_features, self.y_len, 100)
        self.decode_features = tf.concat([decode_features, lstm_decode_features], axis=2)

        return tf.expand_dims(self.x_centered, 2)

    def encode(self, x, features):
        x = tf.concat([x, features], axis=2)

        h = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='x-init',
        )
        c = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='c-init',
        )

        conv_inputs = [h]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)[:-1]):
            dilated_conv = temporal_convolution_layer(
                inputs=h,
                output_units=4*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-encode-{}'.format(i),
            )
            input_gate, conv_filter, conv_gate, emit_gate = tf.split(dilated_conv, 4, axis=2)

            c = tf.nn.sigmoid(input_gate)*c + tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)
            h = tf.nn.sigmoid(emit_gate)*tf.nn.tanh(c)
            conv_inputs.append(h)

        return conv_inputs

    def initialize_decode_params(self, x, features):
        x = tf.concat([x, features], axis=2)

        h = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='h-init-decode',
        )
        c = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='c-init-decode',
        )

        skip_outputs = []
        conv_inputs = [h]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv = temporal_convolution_layer(
                inputs=h,
                output_units=4*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-decode-{}'.format(i),
            )
            input_gate, conv_filter, conv_gate, emit_gate = tf.split(dilated_conv, 4, axis=2)

            c = tf.nn.sigmoid(input_gate)*c + tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)
            h = tf.nn.sigmoid(emit_gate)*tf.nn.tanh(c)

            skip_outputs.append(h)
            conv_inputs.append(h)

        skip_outputs = tf.concat(skip_outputs, axis=2)
        h = time_distributed_dense_layer(skip_outputs, 128, scope='dense-decode-1', activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 2, scope='dense-decode-2')
        return y_hat

    def decode(self, x, conv_inputs, features):
        batch_size = tf.shape(x)[0]

        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(conv_inputs, self.dilations)):
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            batch_idx = tf.reshape(batch_idx, [-1])

            temporal_idx = tf.expand_dims(self.x_len, 1) + tf.expand_dims(tf.range(dilation), 0)
            temporal_idx = tf.reshape(temporal_idx, [-1])

            idx = tf.stack([batch_idx, temporal_idx], axis=1)
            padding = tf.zeros([batch_size, dilation + 1, shape(conv_input, 2)])
            conv_input = tf.concat([padding, conv_input], axis=1)
            slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, shape(conv_input, 2)))

            layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation + self.num_decode_steps)
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        features_ta = tf.TensorArray(dtype=tf.float32, size=self.num_decode_steps)
        features_ta = features_ta.unstack(tf.transpose(features, (1, 0, 2)))

        # initialize output tensor array
        emit_ta = tf.TensorArray(size=self.num_decode_steps, dtype=tf.float32)

        # initialize other loop vars
        elements_finished = 0 >= self.y_len
        time = tf.constant(0, dtype=tf.int32)

        # get initial x input
        current_idx = tf.stack([tf.range(tf.shape(self.x_len)[0]), self.x_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx)

        def loop_fn(time, current_input, queues):
            current_features = features_ta.read(time)
            current_input = tf.concat([current_input, current_features], axis=1)

            with tf.variable_scope('h-init-decode', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                h = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            with tf.variable_scope('c-init-decode', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                c = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            skip_outputs, updated_queues = [], []
            for i, (queue, dilation) in enumerate(zip(queues, self.dilations)):

                state = queue.read(time)
                with tf.variable_scope('dilated-conv-decode-{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights')
                    b_conv = tf.get_variable('biases')
                    dilated_conv = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(h, w_conv[1, :, :]) + b_conv

                input_gate, conv_filter, conv_gate, emit_gate = tf.split(dilated_conv, 4, axis=1)

                c = tf.nn.sigmoid(input_gate)*c + tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)
                h = tf.nn.sigmoid(emit_gate)*tf.nn.tanh(c)

                skip_outputs.append(h)
                updated_queues.append(queue.write(time + dilation, h))

            skip_outputs = tf.concat(skip_outputs, axis=1)
            with tf.variable_scope('dense-decode-1', reuse=True):
                w_h = tf.get_variable('weights')
                b_h = tf.get_variable('biases')
                h = tf.nn.relu(tf.matmul(skip_outputs, w_h) + b_h)

            with tf.variable_scope('dense-decode-2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                y_hat = tf.matmul(h, w_y) + b_y

            elements_finished = (time >= self.y_len)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 2], dtype=tf.float32),
                lambda: y_hat
            )
            next_elements_finished = (time >= self.num_decode_steps - 1)

            return (next_elements_finished, next_input, updated_queues)

        def condition(unused_time, elements_finished, *_):
            return tf.logical_not(tf.reduce_all(elements_finished))

        def body(time, elements_finished, emit_ta, *state_queues):
            (next_finished, emit_output, state_queues) = loop_fn(time, initial_input, state_queues)

            emit = tf.where(elements_finished, tf.zeros_like(emit_output), emit_output)
            emit_ta = emit_ta.write(time, emit)

            elements_finished = tf.logical_or(elements_finished, next_finished)
            return [time + 1, elements_finished, emit_ta] + list(state_queues)

        returned = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[time, elements_finished, emit_ta] + state_queues
        )

        outputs_ta = returned[2]
        y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))
        return y_hat

    def calculate_loss(self):
        x = self.get_input_sequences()

        conv_inputs = self.encode(x, features=self.encode_features)
        decode_x = tf.concat([x, 1.0 - tf.expand_dims(self.x_is_zero, 2)], axis=2)
        self.initialize_decode_params(decode_x, features=self.decode_features)

        y_hat = self.decode(decode_x, conv_inputs, features=self.decode_features)
        y_hat, p = tf.unstack(y_hat, axis=2, num=2)
        y_hat = tf.nn.sigmoid(p)*(y_hat + self.x_mean)
        self.loss = sequence_rmse(self.y, y_hat, self.y_len, weights=self.weights)

        self.prediction_tensors = {
            'preds': tf.nn.relu(y_hat),
            'lengths': self.x_len,
            'ids': self.y_id,
        }

        return self.loss


if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data/processed/'))

    nn = cnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rates=[.001, .0005, .00025],
        beta1_decays=[.9, .9, .9],
        batch_sizes=[64, 128, 256],
        num_training_steps=200000,
        patiences=[5000, 5000, 5000],
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=True,
        min_steps_to_checkpoint=500,
        log_interval=50,
        validation_batch_size=4*64,
        grad_clip=20,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(9)]*3,
        filter_widths=[2 for i in range(9)]*3,
        num_decode_steps=16,
        loss_averaging_window=200
    )
    nn.fit()
    nn.restore()
    nn.predict()
