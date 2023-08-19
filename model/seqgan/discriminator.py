import tensorflow as tf
import numpy as np


def linear(input_, output_size, scope=None):
    '''
        Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
        Args:
        input_: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
      Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
      '''

    # 其实就是一个dense 全链接神经网络
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments : %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx)) # 经过relu后被保留的信息

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias) # 经过sigmoid后被保留的信息，t代表原始信息被转化的比例

            output = t * g + (1. - t) * input_ # 1-t代表留下的 input的信息的比例
            input_ = output

    return output


class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, config):
        self.sequence_length = config.sequence_length
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.filter_sizes = config.dis_filter_sizes # 卷积核的大小
        self.num_filters = config.dis_num_filters # 卷积核的个数
        self.vocab_size = config.vocab_size
        self.dis_learning_rate = config.dis_learning_rate
        self.embedding_size = config.dis_embedding_dim
        self.l2_reg_lambda = config.dis_l2_reg_lambda
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_y') # 类别的one hot编码
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

    def build_discriminator(self):
        with tf.variable_scope('discriminator'):
            with tf.name_scope('embedding'):
                self.W = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size], -1.0, 1.0), name='W')
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # batch * seq * emb_size  embedding层
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  # batch * seq * emb_size * 1
                                                                                        #[batch,seq,emb_size,1]
            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope('conv_maxpool-%s' % filter_size): #
                    filter_shape = [filter_size, self.embedding_size, 1, num_filter] # 卷积核大小input_dim*embedding_size ,input_dim. output_dim/file_num
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name='b') # 生成卷积核的初始权重以及偏置。
                    # print(self.embedded_chars_expanded)
                    conv = tf.nn.conv2d( # 经过embedding之后[batch,seq,emb_size,1],然后经过卷积
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv'
                    ) # [batch , seq_size,1,100]
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), # 然后经过一层relu
                                   name='relu')  # batch * seq - filter_size + 1 * 1 * num_filter
                    pooled = tf.nn.max_pool( # 经过max_pooling
                        h,
                        ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool'
                    )  # batch * 1 * 1 * num_filter # 将整个seq的embedding vector 提取为1个
                    # [N,1,1,100] [N,1,1,200] [N,1,1,200] [N,1,1,200] [N,1,1,200] [N,1,1,100] [N,1,1,100] [N,1,1,100] [N,1,1,100] [N,1,1,100] [N,1,1,160] [N,1,1,160]
                    pooled_outputs.append(pooled) # [N,1,1,output_dim]

            num_filters_total = sum(self.num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3) # 将所有卷积出的第三维的信息拼接到一起 [N,1,1,sum(self.num_filters)]
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # 降维： 【batch * sum_num_fiters】

            with tf.name_scope('highway'):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0) # 经过一次h_highway 激活函数处理

            with tf.name_scope("dropout"): # 经过一层池化层
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            with tf.name_scope("output"): # 进行output
                W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
                self.l2_loss += tf.nn.l2_loss(W) # output 层经过一个L2正则化处理
                self.l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')  # dense 降维 num_filters_total-> num_classes
                self.ypred_for_auc = tf.nn.softmax(self.scores) # 将output class的概率分布进行归一化处理
                self.predictions = tf.argmax(self.scores, 1, name='predictions') # 得到batch_size条预测结果，batch_size条序列分别是真实的还是虚假的

            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y) # 计算batch_size条序列的交叉熵损失
                # 损失函数中加入了正则项
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(self.dis_learning_rate)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
