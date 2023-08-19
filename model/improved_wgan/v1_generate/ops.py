import tensorflow as tf
import numpy as np


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
                        key_masks=None,
                        query_masks=None,
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
    如果使用mask，就不能使用efficient attention加速运算
    Returns
      A 3d tensor with shape of (N, num_steps, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True,
                            kernel_initializer=kernel_initializer)  # (N, num_steps, d_model) # (2,3,,3)
        K = tf.layers.dense(keys, d_model, use_bias=True,
                            kernel_initializer=kernel_initializer)  # (N, num_steps, d_model) # (2,3,,3)
        V = tf.layers.dense(values, d_model, use_bias=True,
                            kernel_initializer=kernel_initializer)  # (N, num_steps, d_model) # 2,3,3

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        num_steps = Q_.get_shape().as_list()[-2]  # seq_size
        head_value_channels = Q_.get_shape().as_list()[-1]
        # Attention
        if key_masks is None and query_masks is None and num_steps > head_value_channels:  # 使用efficient attention加速运算
            outputs = linear_scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training, key_masks,
                                                          query_masks, num_heads=num_heads)
            # outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training, key_masks,
            #                                              query_masks, num_heads=num_heads)
            # # Restore shape (N*h,Q_seq_size, V_dim)
            # outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, Q_seq_size, V_dim*h)
        else:
            outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training, key_masks,
                                                         query_masks, num_heads=num_heads)
            # Restore shape (N*h,Q_seq_size, V_dim)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, Q_seq_size, V_dim*h)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs


def mask(inputs, key_masks=None, query_masks=None, keys=None, type=None):
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
    if type in ("k", "key", "keys") and key_masks is not None:
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0],
                                        1])  # (h*N, seqlen)  # 对key_mask，在第0维复制 head_num遍
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + (1 - key_masks) * padding_num
    elif type in ("q", "query", "queries") and query_masks is not None:  # 将
        # Generate masks
        query_masks = tf.to_float(query_masks)  # (N, T_q)
        query_masks = tf.tile(query_masks, [tf.shape(inputs)[0] // tf.shape(query_masks)[0], 1])  # (h*N, T_q)
        query_masks = tf.expand_dims(query_masks, -1)  # (h*N, T_q, 1)
        query_masks = tf.tile(query_masks, [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        # Apply masks to inputs
        outputs = inputs * query_masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def sinusoidal_position_embeddings(inputs):
    """
    input:[h*N,seq_size,dim]

    """

    seq_len = inputs.get_shape().as_list()[1]
    output_dim = inputs.get_shape().as_list()[2]
    position_ids = tf.range(0, seq_len, dtype=tf.float32)

    indices = tf.range(0, output_dim // 2, dtype=tf.float32)
    indices = tf.pow(10000.0, -2 * indices / output_dim)
    embeddings = tf.einsum('n,d->nd', position_ids, indices)
    embeddings = tf.stack([tf.sin(embeddings), tf.cos(embeddings)],
                          axis=-1)  # [seq_len, output_dim // 2, 2] # 一个二维列表，每一个元素都是[512，32]
    embeddings = tf.reshape(embeddings, [seq_len, output_dim])
    embeddings = embeddings[None, None, :, :]  # [1, 1, seq_len, output_dim]
    return embeddings


def last_dim_repeat_interleave(tensor_v, repeats):
    tensor_shape = tensor_v.get_shape().as_list()
    repeat_tensor_list = []
    for i in range(repeats):
        repeat_tensor_list.append(tensor_v)
    a2 = tf.stack(repeat_tensor_list, -1)

    tensor_shape[-1] = tensor_shape[-1] * repeats
    reshape_shape = tensor_shape
    output = tf.reshape(a2, reshape_shape)
    return output


def relative_position_embedding(Q, K, num_head=None):
    """
    Q:[h*N,seq_size,dim] dim指的是每一个head的dim
    K:[h*N,seq_size,dim]
    """
    dim = Q.get_shape().as_list()[-1]
    seq_size = Q.get_shape().as_list()[-2]
    sinusoidal_positions = sinusoidal_position_embeddings(Q)
    # 计算cos sinusoidal_positions【1，1，512，64】
    cos_pos = last_dim_repeat_interleave(sinusoidal_positions[..., 1::2], 2)
    # 计算sin
    sin_pos = last_dim_repeat_interleave(sinusoidal_positions[..., ::2], 2)

    query_layer = tf.reshape(Q, [-1, num_head, seq_size, dim])
    key_layer = tf.reshape(K, [-1, num_head, seq_size, dim])

    qw2 = tf.stack([-query_layer[..., 1::2], query_layer[..., ::2]], axis=-1)
    qw2 = tf.reshape(qw2, query_layer.get_shape().as_list())
    query_layer = query_layer * cos_pos + qw2 * sin_pos
    query_layer = tf.reshape(query_layer, Q.get_shape().as_list())

    kw2 = tf.stack([-key_layer[..., 1::2], key_layer[..., ::2]], axis=-1)
    kw2 = tf.reshape(kw2, key_layer.get_shape().as_list())
    key_layer = key_layer * cos_pos + kw2 * sin_pos
    key_layer = tf.reshape(key_layer, K.get_shape().as_list())

    return query_layer, key_layer


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 key_masks=None,
                                 query_masks=None,
                                 num_heads=None,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N*h, num_steps, d_model//h].
    K: Packed keys. 3d tensor. [N*h, num_steps, d_model//h].
    V: Packed values. 3d tensor. [N*h, num_steps, d_model//h].
    key_masks: A 2d tensor with shape of [N, num_steps]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        Q, K = relative_position_embedding(Q, K, num_heads) # Q [N*h, num_steps, d_model//h] | K [N*h, num_steps, d_model//h]
        # dot product    Q(batch_size,seq_size,dim)         K(batch_size,dim , seq_size)
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) # [N*h,Q_num_steps,K_num_steps]

        # scale
        outputs /= d_k ** 0.5

        # key masking
        if key_masks is not None:
            outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)  # (h*N,T_q,T_k)

        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        if query_masks is not None:
            outputs = mask(outputs, query_masks=query_masks, keys=K, type="query")
        # attention1 = outputs
        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)  [N*h,Q_num_steps,K_num_steps] V [N*h,V_num_steps,V_dim]
        outputs = tf.matmul(outputs, V)  # (N*h,Q_seq_size, V_dim)

    return outputs


def linear_scaled_dot_product_attention(Q, K, V,
                                        causality=False, dropout_rate=0.,
                                        training=True,
                                        key_masks=None,
                                        query_masks=None,
                                        num_heads=None,
                                        scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N*h, num_steps, d_model//h].
    K: Packed keys. 3d tensor. [N*h, num_steps, d_model//h].
    V: Packed values. 3d tensor. [N*h, num_steps, d_model//h].
    key_masks: A 2d tensor with shape of [N, num_steps]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        H_C,seq_size,d_k = Q.get_shape().as_list()
        Q, K = relative_position_embedding(Q, K, num_heads)

        # Split and concat
        Q = tf.concat(tf.split(Q, num_heads, axis=0), axis=2)  # (N, T_q, H*dim)
        K = tf.concat(tf.split(K, num_heads, axis=0), axis=2)  # (N, T_k, H*dim)
        V = tf.concat(tf.split(V, num_heads, axis=0), axis=2)  # (N, T_k, H*dim)


        Q = tf.transpose(Q, [0, 2, 1]) # (N, H*dim, seq_size)
        K = tf.transpose(K, [0, 2, 1])  # (N, H*dim, seq_size)
        V = tf.transpose(V, [0, 2, 1])  # (N, H*dim, seq_size)
        # Q = tf.reshape(Q,[-1,d_k*num_heads,seq_size]) # (N,H*dim,seq_size)
        # K = tf.reshape(K, [-1, d_k * num_heads, seq_size]) # (N,H*dim,seq_size)
        # V = tf.reshape(V, [-1, d_k * num_heads, seq_size])  # (N,H*dim,seq_size)

        head_value_channels = d_k
        attended_values = []
        for i in range(num_heads):
            key = tf.nn.softmax(
                K[:,i * head_value_channels:(i + 1) * head_value_channels, :],
                axis=2
            )
            query = tf.nn.softmax(
                Q[:,i * head_value_channels:(i + 1) * head_value_channels, :],
                axis=1
            )
            value = V[:,i * head_value_channels:(i + 1) * head_value_channels, :]
            context = tf.matmul(key,tf.transpose(value, [0, 2, 1]))  # (batch_size,k_dim,V_dim)
            # context /= d_k ** 0.5 # 进行放缩
            # # dropout
            # context = tf.layers.dropout(context, rate=dropout_rate, training=training) # dropoutput
            attended_value = tf.matmul(tf.transpose(context, [0, 2, 1]),
                                       query, [0, 2, 1])  # (batch_size,V_dim,seq_size)

            attended_values.append(attended_value)
        aggregated_values = tf.concat(attended_values, axis=1)  # (batch_size,head_num * V_dim,seq_size)
        aggregated_values = tf.transpose(aggregated_values, [0, 2, 1]) # (batch_size,seq_size,head_num * V_dim)
        # aggregated_values = tf.concat(tf.split(aggregated_values, num_heads, axis=2),
        #                               axis=0)  # (head_num*batch_size,V_dim,seq_size)


    return aggregated_values


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
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.selu,
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

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf