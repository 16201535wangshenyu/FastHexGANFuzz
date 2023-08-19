import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


def positional_encoding(enc, num_steps, scope="positional_encoding"):
    '''
    Sinusoidal Positional_Encoding.
    :param enc:
    :param num_steps: scalar
    :param scope:
    :return:
    '''

    E = enc.get_shape().as_list()[-1]
    N, T = tf.shape(enc)[0], tf.shape(enc)[1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # postion indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(num_steps)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (num_steps, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        return tf.to_float(outputs)


def multihead_attention(queries, keys, values, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, num_steps, d_model].
    keys: A 3d tensor with shape of [N, num_steps, d_model].
    values: A 3d tensor with shape of [N, num_steps, d_model].
    key_masks: A 2d tensor with shape of [N, num_steps]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, num_steps, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True,
                            kernel_initializer=kernel_initializer)  # (N, num_steps, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True,
                            kernel_initializer=kernel_initializer)  # (N, num_steps, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True,
                            kernel_initializer=kernel_initializer)  # (N, num_steps, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs


def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    # elif type in ("q", "query", "queries"):
    #     # Generate masks
    #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    #
    #     # Apply masks to inputs
    #     outputs = inputs*masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, num_steps, d_model].
    K: Packed keys. 3d tensor. [N, num_steps, d_model].
    V: Packed values. 3d tensor. [N, num_steps, d_model].
    key_masks: A 2d tensor with shape of [N, num_steps]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

        # scale
        outputs /= d_k ** 0.5

        # key masking
        # outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1],
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs


def relu(x):
    return tf.nn.relu(x)


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


##################################BLSTM#########################################

def _witch_cell(cell_type, hidden_unit, is_trainable=True, initializer=None):
    """
    RNN 类型,先创建出每个cell的类型
    :return:
    """
    cell_tmp = None
    if cell_type == 'lstm':  # hidden_unit LSTM cell的宽度
        cell_tmp = rnn.LSTMCell(num_units=hidden_unit, initializer=initializer)
    elif cell_type == 'gru':
        cell_tmp = rnn.GRUCell(num_units=hidden_unit, kernel_initializer=initializer, bias_initializer=initializer)
    return cell_tmp


def _bi_dir_rnn(cell_type, hidden_unit, dropout_rate, is_trainable=True, initializer=None):
    """
    双向RNN：每个cell在输出的时候进行dropout
    :return:
    """
    cell_fw = _witch_cell(cell_type=cell_type, hidden_unit=hidden_unit, is_trainable=is_trainable, initializer=initializer)  # 建立一个前向的LSTM
    cell_bw = _witch_cell(cell_type=cell_type, hidden_unit=hidden_unit, is_trainable=is_trainable, initializer=initializer)  # 建立一个后向的LSTM
    # if dropout_rate is not None and is_trainable:  #
    #     cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_rate)
    #     cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_rate)
    return cell_fw, cell_bw


def blstm_layer(cell_type, hidden_unit, dropout_rate, embedding_chars, num_layers, is_trainable=True, initializer=None):
    """
使用多个cell扩展成一个lstm层
两个lstm层组装成blstm
    :return:
    """
    with tf.variable_scope('rnn_layer'):
        cell_fw, cell_bw = _bi_dir_rnn(cell_type=cell_type, hidden_unit=hidden_unit, dropout_rate=dropout_rate, is_trainable=is_trainable,
                                       initializer=initializer)
        if num_layers > 1:
            cell_fw = rnn.MultiRNNCell([cell_fw] * num_layers, state_is_tuple=True)
            cell_bw = rnn.MultiRNNCell([cell_bw] * num_layers, state_is_tuple=True)
        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
        #                                              dtype=tf.float32)
        sequence = tf.unstack(embedding_chars, axis=1)
        hs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw, cell_bw,sequence,dtype=tf.float32)
        hs = tf.stack(
            values=hs,
            axis=1)
        output = tf.reduce_sum(
            tf.reshape(hs, shape=[-1, hs.shape[1], 2, hidden_unit]),
            axis=2
        )
        # output_fw, output_bw = outputs
        # outputs = output_fw + output_bw
        if output is None:
            print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
        if dropout_rate is not None and is_trainable:
            output = tf.layers.dropout(output,rate = dropout_rate,name="dropout")

    return output


def project_bilstm_layer(hidden_unit, initializers, seq_length, num_labels, lstm_outputs, name=None, is_trainable=True):
    """
处理blstm的输出，先经过一个全连接层，然后经过一个logits
    hidden layer between lstm layer and logits
    :param lstm_outputs: [batch_size, num_steps, emb_size]
    :return: [batch_size, num_steps, num_tags]
    """
    with tf.variable_scope("project" if not name else name):
        with tf.variable_scope("hidden"):
            W = tf.get_variable("W", shape=[hidden_unit * hidden_unit, hidden_unit],
                                dtype=tf.float32, initializer=initializers, trainable=is_trainable)
            b = tf.get_variable("b", shape=[hidden_unit], dtype=tf.float32,
                                initializer=tf.zeros_initializer(), trainable=is_trainable)
            output = tf.reshape(lstm_outputs, shape=[-1, hidden_unit * hidden_unit])
            hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
        # project to score of tags
        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[hidden_unit, seq_length*num_labels],
                                dtype=tf.float32, initializer=initializers, trainable=is_trainable)
            b = tf.get_variable("b", shape=[seq_length*num_labels], dtype=tf.float32,
                                initializer=tf.zeros_initializer(), trainable=is_trainable)
            pred = tf.nn.xw_plus_b(hidden, W, b)
            output = tf.reshape(pred, [-1, seq_length, num_labels])
            # output = tf.nn.softmax(output, name="softmax")

        return output


def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma

    return x
