from datetime import timedelta

import tensorflow as tf
import time

import os
import blstm_dcgan_tflib as lib
import blstm_dcgan_tflib.ops.linear
# import tflib.ops.conv1d
import blstm_dcgan_tflib.ops.conv2d
import blstm_dcgan_tflib.ops.deconv2d
import blstm_dcgan_tflib.ops.batchnorm
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow.contrib as tc
from utils import *
import ops as ops

he_init = tf.contrib.layers.variance_scaling_initializer()
xavier_init = tf.contrib.layers.xavier_initializer()
normal_init = tf.random_normal_initializer(mean=0, stddev=0.02)


class BLSTM_DCGAN(object):
    def __init__(self, sess, args, w2i, i2w):
        lib.delete_all_params()
        lib.delete_param_aliases()
        self.model_name = "BLSTM_DCGAN_SEQ"
        self.sess = sess

        self.w2i = w2i
        self.i2w = i2w
        self.outputdir = args.outputdir
        if not os.path.exists(self.outputdir):  # os模块判断并创建
            os.mkdir(self.outputdir)

        self.LAMBDA = args.LAMBDA  # 梯度惩罚
        self.epoch = args.epoch  # 一个epoch将所有的训练数据样本训练一次
        self.disc_pre_train_epoch = args.disc_pre_train_epoch
        self.BLSTM_pre_train_epoch = args.BLSTM_pre_train_epoch
        self.critic_iters = args.critic_iters  # 训练，鉴别器需要迭代的次数
        self.batch_size = args.batch_size  #
        self.seq_size = args.seq_size  # 报文x的长度
        self.embedding_size = self.seq_size
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.num_blocks = args.num_blocks
        self.d_ff = args.d_ff  # 前馈层参数

        self.z_dim = args.z_dim  # 噪声data的维数 50

        self.vocab_size = args.vocab_size

        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.b_learning_rate = args.b_lr

        self.g_beta1 = args.g_beta1
        self.g_beta2 = args.g_beta2

        self.d_beta1 = args.d_beta1
        self.d_beta2 = args.d_beta2

        self.b_beta1 = args.b_beta1
        self.b_beta2 = args.b_beta2

        self.d_model = args.d_model  # embedding_dim

        # self.data = load_data(args.data_file, self.w2i)# [total_num * seq_size]

        # self.embeddings = tf.get_variable('weight_mat',
        #                                   dtype=tf.float32,
        #                                   shape=(self.vocab_size, self.d_model),
        #                                   initializer=tf.contrib.layers.xavier_initializer())

        #################################

        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
        self.real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_size])
        self.real_inputs = tf.one_hot(self.real_inputs_discrete, self.vocab_size)
        _, self.fake_inputs = self.BLSTM_Layer(self.generator(self.z),is_training=False,
                                               is_real_frame=False)  # shape (batch_size, seq_size, vocab_size), there is probablity
        self.fake_inputs_discrete = tf.argmax(self.fake_inputs, self.fake_inputs.get_shape().ndims - 1)

        _, self.fake_inputs_test = self.BLSTM_Layer(self.generator(self.z, False), is_real_frame=False,
                                                    is_training=False)
        self.fake_inputs_discrete_test = tf.argmax(self.fake_inputs_test, self.fake_inputs_test.get_shape().ndims - 1)
       
        ######################################################
        _, self.fake_test = self.BLSTM_Layer(tf.random_normal([self.batch_size,1,256,256]), is_real_frame=False,
                                                    is_training=False)
        self.fake_test_discrete = tf.argmax(self.fake_test,self.fake_test.get_shape().ndims - 1)


        #########################################################

        self.real_blstm_hidden_output, self.real_blstm_logits = self.BLSTM_Layer(self.real_inputs,is_training=True)

        ######################################################
        self.real_test_discrete = tf.argmax(self.BLSTM_Layer(self.real_inputs,is_training=False)[1],self.real_blstm_logits.get_shape().ndims - 1)
        self.generator_output = self.generator(self.z)[0][0]
        self.BLSTM_output = self.BLSTM_Layer(self.real_inputs,is_training=False)[0][0][0]
        #########################################################

        self.real_logits = self.discriminator(self.BLSTM_Layer(self.real_inputs,is_training=False)[0])
        self.fake_logits = self.discriminator(self.generator(self.z))

        # self.gen_cost = -tf.reduce_mean(tf.log(self.fake_logits)) tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,labels=tf.zeros_like(self.fake_logits))
        # self.disc_cost = tf.reduce_mean(tf.log(self.fake_logits)) - tf.reduce_mean(tf.log(self.real_logits))
        self.gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,labels=tf.ones_like(self.fake_logits)))
        self.disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,labels=self.smooth_neg_labels(tf.zeros_like(self.fake_logits))))            
        self.disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits,labels=self.smooth_pos_labels(tf.ones_like(self.real_logits)))) 
        
        # self.blstm_cost = -tf.reduce_sum(self.real_inputs * tf.log(self.real_blstm_logits))
        self.blstm_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.real_blstm_logits,labels=self.real_inputs))

        self.w_distance = tf.reduce_mean(self.real_logits) - tf.reduce_mean(self.fake_logits)

        self.gen_prob = tf.reduce_mean(tf.sigmoid(self.fake_logits))
        self.real_prob = tf.reduce_mean(tf.sigmoid(self.real_logits))

        # improved WGAN lipschitz-penalty

        self.gradient_penalty = tf.reduce_mean(tf.abs(self.real_blstm_hidden_output - self.generator(self.z))) ##本质上reduce sum与reduce_mean最小化的效果是一样的，为了统一度量，改为readuce_mean

        # self.disc_cost += self.LAMBDA * self.gradient_penalty

        # self.reg = tc.layers.apply_regularization(
        #     tc.layers.l1_regularizer(2.5e-4),
        #     weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        # )
        self.blstm_reg = tc.layers.apply_regularization(
            tc.layers.l2_regularizer(0.9),
            weights_list=[var for var in tf.global_variables() if ('project' in var.name and "W" in var.name) or ('rnn_layer' in var.name and "kernel" in var.name)]
        )
        # self.g_loss_reg = self.gen_cost + self.reg
        self.b_loss_reg = self.blstm_cost + self.blstm_reg
        # self.d_loss_reg = self.disc_cost + self.reg
        self.g_loss_reg = self.gen_cost  # 原文并没有说使用了正则化
        self.d_loss_reg = self.disc_cost + self.LAMBDA * self.gradient_penalty

        self.variable = tf.trainable_variables()
        self.gen_params = [v for v in self.variable if 'Generator' in v.op.name]
        self.disc_params = [v for v in self.variable if 'discriminator' in v.op.name]
        self.blstm_params = [v for v in self.variable if 'BLSTM_layer' in v.op.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=self.g_beta1,
                                                       beta2=self.g_beta2).minimize(self.g_loss_reg,
                                                                                    var_list=self.gen_params,
                                                                                    colocate_gradients_with_ops=True)
            self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=self.d_beta1,
                                                        beta2=self.d_beta2).minimize(self.d_loss_reg,
                                                                                     var_list=self.disc_params,
                                                                                     colocate_gradients_with_ops=True)
            self.blstm_train_op = tf.train.AdamOptimizer(learning_rate=self.b_learning_rate, beta1=self.b_beta1,
                                                         beta2=self.b_beta2).minimize(self.b_loss_reg,
                                                                                      var_list=self.blstm_params,
                                                                                      colocate_gradients_with_ops=True)

            # self.disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4) \
            #     .minimize(self.d_loss_reg, var_list=self.disc_params)
            # self.gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4) \
            #     .minimize(self.g_loss_reg, var_list=self.gen_params)

        # weight clipping
        # self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.disc_params]

        self.d_sum = tf.summary.scalar("gen_cost", self.gen_cost)
        self.g_sum = tf.summary.scalar("disc_cost", self.disc_cost)



    def smooth_pos_labels(self,y):
        """
        https://jiuaidu.com/jianzhan/980986/
        使得true label的值的范围为[0.8,1]
        :param y:
        :return:
        """
        return y - (tf.random_uniform(y.get_shape().as_list()) * 0.1)
    def smooth_neg_labels(self,y):
        """
        使得fake label的值的范围为[0.0,0.3]
        :param y:
        :return:
        """
        return y + tf.random_uniform(y.get_shape().as_list()) * 0.1


    def generator(self, z, is_training=True):
        """
        achitecture:

        [one] linear layer
        [num_blocks] transformer block
        [one] linear layer
        [one] softmax layer
        """
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):  # z's shape (batch_size, z_dim)

            # output = lib.ops.linear.Linear('Generator.Input', self.z_dim, 4 * 4 * 4 * self.d_model, z,
            #                                initializer=None)  # 没有使用normal_init
            output = tf.layers.dense(z, units=4 * 4 * 4 * self.d_model, kernel_initializer=normal_init,name='Generator.Input')
            # output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4 * self.d_model, 4, 4])
            # result_txt = open("/home/shenyuwang/paper_fuzzing/workspace/model/blstm_DCGAN/model/v1/tmp_var/result.txt","a")
            # print("output:",output.get_shape(),file = result_txt)
            output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0, 2, 3], output, fused=True)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * self.d_model, 4 * self.d_model, 5, output,
                                               initializer=None) # [512,4,4]->[256,8,8]
            output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output, fused=True)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.3', 4 * self.d_model, 2 * self.d_model, 5, output,
                                               initializer=None) # [256,8,8]->[128,16,16]
            output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output, fused=True)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.4', 2 * self.d_model, 2 * self.d_model, 5, output,
                                               initializer=None)  # [128,16,16]->[64,32,32]
            output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0, 2, 3], output, fused=True)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.5', 2 * self.d_model, self.d_model, 5, output,
                                               initializer=None)  # [64,32,32]->[1,64,64]
            output = lib.ops.batchnorm.Batchnorm('Generator.BN5', [0, 2, 3], output, fused=True)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.6', 1 * self.d_model, 1 * self.d_model, 5, output,
                                               initializer=None)
            # output = lib.ops.batchnorm.Batchnorm('Generator.BN6', [0, 2, 3], output, fused=True)
            output6 = tf.nn.relu(output)

            output7 = lib.ops.deconv2d.Deconv2D('Generator.7', self.d_model, 1, 5, output6, initializer=None)
            output = tf.tanh(output7)


            lib.ops.conv2d.unset_weights_stdev()
            lib.ops.deconv2d.unset_weights_stdev()
            lib.ops.linear.unset_weights_stdev()

        return output

    def BLSTM_Layer(self, x, is_real_frame=True, is_training=True):
        blstm_hidden_output = None

        with tf.variable_scope("BLSTM_layer", reuse=tf.AUTO_REUSE):
            if is_real_frame:
                # embedding layer
                x = tf.argmax(x, x.get_shape().ndims - 1)
                x = tf.cast(x, dtype=tf.int64)
                output = tf.contrib.layers.embed_sequence(x, self.vocab_size, self.embedding_size,
                                                          trainable=is_training)
                # output = tf.cast(output, dtype=tf.float32)
                # shape: [batch_size, seq_len, embedding_size]
                # output = tf.nn.dropout(embed_seq, keep_prob=self.dropout_rate)
                                           
                output = ops.blstm_layer("lstm", self.seq_size, self.dropout_rate, output, 1, is_trainable=is_training,
                                         initializer=normal_init)
                blstm_output = ops.blstm_layer("lstm", self.seq_size, None, output, 1, is_trainable=is_training,
                                               initializer=normal_init)
                blstm_hidden_output = tf.reshape(blstm_output, [-1, 1, self.seq_size, self.seq_size])
            else:
                blstm_output = tf.reshape(x, [-1, self.seq_size, self.seq_size])

            # output = lib.ops.linear.Linear('output_layer.dense1',  self.seq_size, self.seq_size, blstm_output,initializer=None)
            # output = lib.ops.linear.Linear('output_layer.dense2',  self.seq_size, self.vocab_size, output,initializer=None)
            with tf.variable_scope('output_layer'):
                output = tf.layers.dense(blstm_output,self.seq_size, kernel_initializer=normal_init,trainable=True,name="output_layer.dense1")
                output = tf.layers.dense(output,self.vocab_size, kernel_initializer=normal_init,trainable=True,name="output_layer.dense2")
            if is_training:
                output = tf.layers.dropout(output,rate = self.dropout_rate,name="output_layer.dropout")
            # output = ops.project_bilstm_layer(self.seq_size, initializers=normal_init, seq_length=self.seq_size,
            #                                   num_labels=self.vocab_size, lstm_outputs=blstm_output,
            #                                   name=None, is_trainable=is_training)

            return blstm_hidden_output, output

    def discriminator(self, x, is_training=True):

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            # 参考 https://github.com/yxtay/char-rnn-text-generation/blob/master/tf_model.py
            # output = tf.transpose(x, [0, 2, 1])  # (batch_size, vocab_size, seq_len)
            # output = lib.ops.conv1d.Conv1D('Conv1d.1', self.vocab_size, self.vocab_size, 5, output)
            # output = lib.ops.conv1d.Conv1D('Conv1d.2', self.vocab_size, self.vocab_size, 5, output)
            # output = lib.ops.conv1d.Conv1D('Conv1d.3', self.vocab_size, self.vocab_size, 5, output)
            # output = tf.transpose(output, [0, 2, 1])
            # output = self.self_attention(output, is_training)
            # output = tf.transpose(output, [0, 2, 1])
            # output = lib.ops.conv1d.Conv1D('Conv1d.4', self.vocab_size, self.vocab_size, 5, output)
            # output = tf.transpose(output, [0, 2, 1])
            # output = self.self_attention(output, is_training)
            # enc = tf.reshape(output, [self.batch_size, self.vocab_size * self.seq_size])
            # res = tf.layers.dense(enc, units=1)
            output = ops.Z_ScoreNormalization(x, 0, 1)
            # output = tf.transpose(output, [0, 2, 3, 1])
            short_cut = lib.ops.conv2d.Conv2D('short_cut.1', 1 , 1, 5, output, stride=2,
                                               initializer=None)
            short_cut = lib.ops.conv2d.Conv2D('short_cut.2', 1 , 1, 5, short_cut, stride=2,
                                               initializer=None) 
            short_cut = tf.reshape(short_cut, [self.batch_size, self.d_model // 2,self.d_model // 2])                                       
            with tf.variable_scope("CNN_layer", reuse=tf.AUTO_REUSE):
                output = lib.ops.conv2d.Conv2D('CNN_layer.1', 1, self.d_model // 2, 5, output, stride=2,
                                               initializer=None) # [1,64,64] -> [64,32,32]
                output = ops.LeakyReLU(output)

                output = lib.ops.conv2d.Conv2D('CNN_layer.2', self.d_model // 2, self.d_model // 2 , 5, output, stride=2,
                                               initializer=None) # [64,32,32] -> [128,16,16]
                output = lib.ops.batchnorm.Batchnorm('CNN_layer.BN2', [0, 2, 3], output)
                output = ops.LeakyReLU(output)

                output = lib.ops.conv2d.Conv2D('CNN_layer.3', self.d_model // 2 , self.d_model, 5, output, stride=2,
                                               initializer=None) # [128,16,16] -> [256,8,8]
                output = lib.ops.batchnorm.Batchnorm('CNN_layer.BN3', [0, 2, 3], output)
                output = ops.LeakyReLU(output)

                output = lib.ops.conv2d.Conv2D('CNN_layer.4',self.d_model, self.d_model, 5, output, stride=2,
                                               initializer=None)# [256,8,8] -> [512,4,4]
                output = lib.ops.batchnorm.Batchnorm('CNN_layer.BN4', [0, 2, 3], output)
                output = ops.LeakyReLU(output)

                output = lib.ops.conv2d.Conv2D('CNN_layer.5', self.d_model, 2 * self.d_model, 5, output, stride=2,
                                               initializer=None)
                output = lib.ops.batchnorm.Batchnorm('CNN_layer.BN5', [0, 2, 3], output)
                output = ops.LeakyReLU(output)

                output = lib.ops.conv2d.Conv2D('CNN_layer.6', 2 * self.d_model, 2 * self.d_model, 5, output, stride=2,
                                               initializer=None)
                output = lib.ops.batchnorm.Batchnorm('CNN_layer.BN6', [0, 2, 3], output)
                output = ops.LeakyReLU(output)

                # output = tf.reshape(output, [-1, 2, self.d_model // 2, self.d_model // 2]) # 【1,64,64】 [512,4,4]
                output = tf.reshape(output, [self.batch_size,self.d_model // 2, self.d_model // 2])
               
                output = output + short_cut

                output = tf.reshape(output, [self.batch_size, (self.d_model // 2) * (self.d_model // 2)])
                # output = lib.ops.linear.Linear('Discriminator.Output', (self.d_model // 2) * (self.d_model // 2), 1, output,
                #                                initializer=None)
                output = tf.layers.dense(output, units=1, kernel_initializer=normal_init,name='Discriminator.Output')
                # output = tf.sigmoid(output)

                lib.ops.conv2d.unset_weights_stdev()
                lib.ops.deconv2d.unset_weights_stdev()
                lib.ops.linear.unset_weights_stdev()

                return tf.reshape(output, [-1])

    # @property
    # def vars(self):
    #     return [var for var in tf.global_variables() if self.name in var.name]

    # def ResBlock(self, name, inputs):
    #     output = inputs
    #     output = tf.nn.relu(output)
    #     output = lib.ops.conv1d.Conv1D(name + '.1', self.d_model, self.d_model, 5, output)
    #     output = tf.nn.relu(output)
    #     output = lib.ops.conv1d.Conv1D(name + '.2', self.d_model, self.d_model, 5, output)
    #     return inputs + (0.3 * output)

    def train(self, data):
        batch = 0
        step = 0  # 代表参数的更新次数
        # epoch_need_record = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        n_batch = len(data) // self.batch_size  # 总batch数
        total_batch = n_batch // 100 * self.epoch  ## 每隔100个batch 记录一次，因此一共记录（n_batch // 100） * self_epoch次
        print("n_batch:", n_batch)
        print("total_batch", n_batch // 100 * self.epoch)
        self.sess.run(tf.global_variables_initializer())
        # self.load(r"/home/shenyuwang/paper_fuzzing/workspace/output/coap/blstm_dcgan/test20230420_19_59_57_coap_116_118_120_122_6.0w/pre_train_blstm_epoch50")
        

        # 画图相关
        ###
        fig_w_distance = np.zeros([total_batch])
        fig_d_loss_trains = np.zeros([total_batch])
        fig_g_loss_trains = np.zeros([total_batch])
        fig_b_loss_trains = np.zeros([total_batch])
        fig_time = np.zeros([total_batch])

        ######################################## 预训练BLSTM ##########################################################
        train_start_time = time.time()
        for e in range(self.BLSTM_pre_train_epoch):
            real_data = inf_train_gen(data, self.batch_size)
            epoch_start_time = time.time()
            iter = 0
            BLSTM_pre_train_epoch_over = False
            while not BLSTM_pre_train_epoch_over:
                z = make_noise(shape=[self.batch_size, self.z_dim])
                real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                _blstm_cost, _ = self.sess.run(
                    [self.blstm_cost, self.blstm_train_op],
                    feed_dict={self.real_inputs_discrete: real_inputs_discrete})
                # print("_blstm_cost:",_blstm_cost)
                # print("_disc_cost:",_disc_cost)
                iter += 1
                step += 1
                # print("current_step:{}".format(step))
                if iter % 100 == 99:
                    iter += 1  # 此batch data 用于测试，而不更新参数，因此step不会自增
                    z = make_noise(shape=[self.batch_size, self.z_dim])
                    real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                    _blstm_cost, = self.sess.run(
                        [self.blstm_cost],
                        feed_dict={self.real_inputs_discrete: real_inputs_discrete})
                    # fig_w_distance[batch] = _w_distance
                    # fig_d_loss_trains[batch] = _disc_cost
                    
                    fig_b_loss_trains[batch] = _blstm_cost
                    fig_time[batch] = time.time() - train_start_time
                    batch += 1
                    print(
                        "Epoch {}\niter {}\n_blstm_cost {} \ntime{}"
                            .format(e + 1, iter,_blstm_cost,
                                    timedelta(seconds=time.time() - epoch_start_time)))
                if iter % n_batch == 0:  ## 每一次训练集中的数据训练完一遍的时候
                    # 某些epoch存checkpoint并且保存生成的数据
                    if (e + 1) % 10 == 0:  # 每隔2个epoch保存一次model
                        self.save_model(e, 'pre_train_blstm_epoch' + str(e + 1), step, train_start_time, batch,
                                        fig_w_distance,
                                        fig_d_loss_trains, fig_g_loss_trains, fig_b_loss_trains, fig_time,
                                        is_pretrain=True)
                #     def save_model(self, e, folder, step, train_start_time, batch, fig_w_distance, fig_d_loss_trains, fig_g_loss_trains,
                #    fig_b_loss_trains,
                #    fig_time, is_pretrain=False)
                    BLSTM_pre_train_epoch_over = True
                    break
        
        ########################################预训练鉴别器#############################################################
        step = 0
        for e in range(self.disc_pre_train_epoch):
            real_data = inf_train_gen(data, self.batch_size)
            epoch_start_time = time.time()
            iter = 0
            disc_pre_train_epoch_over = False
            while not disc_pre_train_epoch_over:
                z = make_noise(shape=[self.batch_size, self.z_dim])
                real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                _disc_cost, _ = self.sess.run(
                    [self.disc_cost, self.disc_train_op],
                    feed_dict={
                        self.real_inputs_discrete: real_inputs_discrete,
                        self.z: z
                        }
                    )

                iter += 1
                step += 1
                # print("current_step:{}".format(step))
                if iter % 100 == 99:
                    iter += 1  # 此batch data 用于测试，而不更新参数，因此step不会自增
                    z = make_noise(shape=[self.batch_size, self.z_dim])
                    real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                    _disc_cost, = self.sess.run(
                        [self.disc_cost],
                        feed_dict={
                            self.real_inputs_discrete: real_inputs_discrete,
                            self.z: z
                            }
                        )
                    # fig_w_distance[batch] = _w_distance
                    # fig_d_loss_trains[batch] = _disc_cost
                    
                    print(
                        "Epoch {}\niter {}\_disc_cost {} \ntime{}"
                            .format(e + 1, iter,_disc_cost,
                                    timedelta(seconds=time.time() - epoch_start_time)))
                if iter % n_batch == 0:  ## 每一次训练集中的数据训练完一遍的时候
                    # 某些epoch存checkpoint并且保存生成的数据
                    # if (e + 1) % 1 == 0:  # 每隔2个epoch保存一次model
                    #     self.save_model(e, 'pre_train_epoch' + str(e + 1), step, train_start_time, batch,
                    #                     fig_w_distance,
                    #                     fig_d_loss_trains, fig_g_loss_trains, fig_b_loss_trains, fig_time,
                    #                     is_pretrain=True)
               
                    disc_pre_train_epoch_over = True
                    break        
        ######################################## 正式训练 ###############################################################
        fig_w_distance = np.zeros([total_batch])
        fig_d_loss_trains = np.zeros([total_batch])
        fig_b_loss_trains = np.zeros([total_batch])
        fig_g_loss_trains = np.zeros([total_batch])
        fig_time = np.zeros([total_batch])
        batch = 0
        step = 0  # 代表参数的更新次数
        train_start_time = time.time()
        for e in range(self.epoch):
            real_data = inf_train_gen(data, self.batch_size)
            epoch_start_time = time.time()
            iter = 0
            epoch_over = False

            while not epoch_over:
                ## 然后训练鉴别器
                for _ in range(self.critic_iters):
                    iter += 1
                    step += 1
                    real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                    z = make_noise([self.batch_size, self.z_dim])
                    # self.sess.run(self.d_clip)
                    _blstm_cost, _disc_cost, _ = self.sess.run(
                        [self.blstm_cost, self.disc_cost, self.disc_train_op],
                        feed_dict={self.real_inputs_discrete: real_inputs_discrete,
                                   self.z: z})
                    if iter % 100 == 99:
                        z = make_noise([self.batch_size, self.z_dim])
                        iter += 1
                        real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                        gen_samples, gen_cost, blstm_cost, disc_cost, gen_prob, real_prob, w_distance = \
                            self.sess.run(
                                [self.fake_inputs_discrete_test, self.gen_cost, self.blstm_cost, self.disc_cost,
                                 self.gen_prob,
                                 self.real_prob, self.w_distance],
                                feed_dict={self.real_inputs_discrete: real_inputs_discrete, self.z: z})
                        fig_w_distance[batch] = w_distance
                        fig_d_loss_trains[batch] = disc_cost
                        fig_b_loss_trains[batch] = blstm_cost
                        fig_g_loss_trains[batch] = gen_cost
                        fig_time[batch] = time.time() - train_start_time
                        batch += 1

                        translate(gen_samples, self.i2w)
                        time.sleep(0.1)
                        print(
                            "Epoch {}\niter {}\ndisc cost {}, real prob {}\ngen cost {}, gen prob {}\nblstm_cost {}\nw-distance {}\ntime{}"
                                .format(e + 1, iter, disc_cost, real_prob, gen_cost, gen_prob, blstm_cost, w_distance,
                                        timedelta(seconds=time.time() - epoch_start_time)))

                    # print("iter:", iter)

                    if iter % n_batch == 0:  ## 每一次训练集中的数据训练完一遍的时候

                        # 某些epoch存checkpoint并且保存生成的数据
                        if (e + 1) % 10 == 0:
                            self.save_model(e, 'epoch' + str(e + 1), step, train_start_time, batch, fig_w_distance,
                                            fig_d_loss_trains, fig_g_loss_trains, fig_b_loss_trains, fig_time,
                                            is_pretrain=False)

                        epoch_over = True
                        break
                ## 先训练的是生成器
                z = make_noise(shape=[self.batch_size, self.z_dim])
                _gen_cost, _ = self.sess.run([self.gen_cost, self.gen_train_op], feed_dict={self.z: z})
                
                z = make_noise(shape=[self.batch_size, self.z_dim])
                _gen_cost, _ = self.sess.run([self.gen_cost, self.gen_train_op], feed_dict={self.z: z})


        # write_to_content = "n_batch:" + str(n_batch) + "\n"
        # write_to_content += "total_batch:" + str(total_batch) + "\n"
        # write_to_content += "After training, total time:" + str(
        #     timedelta(seconds=time.time() - train_start_time)) + "\n"
        # write_to_content += "Train epoch:" + str(self.epoch) + "\n"
        # self.save_final_info("result.txt", write_to_content)

        print("After training, total time:", timedelta(seconds=time.time() - train_start_time))
        print("w_sitance")
        print(fig_w_distance)
        print("\nd_loss")
        print(fig_d_loss_trains)
        print("\ng_loss")
        print(fig_g_loss_trains)
        print("\ntime")
        print(fig_time)

        lib.delete_all_params()
        lib.delete_param_aliases()
# self.draw_result_picture(batch, fig_w_distance_t, fig_d_loss_trains_t, fig_g_loss_trains_t, fig_b_loss_trains_t,
#                                  folder, fig_time_t, is_pretrain)
    def draw_result_picture(self, total_batch, fig_w_distance, fig_d_loss_trains, fig_g_loss_trains, fig_b_loss_trains,
                            folder,
                            fig_time, is_pretrain=False):
        save_dir = self.outputdir + os.path.sep + folder
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns = None

        lns1 = ax1.plot(np.arange(total_batch), fig_w_distance, label="w_distance")
        lns2 = ax2.plot(np.arange(total_batch), fig_d_loss_trains, 'r', label="Critic_loss")
        lns3 = ax2.plot(np.arange(total_batch), fig_g_loss_trains, 'g', label="Generator_loss")
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('w_distance')

        if not is_pretrain:
            ax2.set_ylabel('Critic_loss & Generator_loss & Blstm_loss')
            # 合并图例
            lns4 = ax2.plot(np.arange(total_batch), fig_b_loss_trains, 'tab:orange', label="Blstm_loss")
            lns = lns1 + lns2 + lns3 + lns4
            labels = ["w_distance", "Critic_loss", "Generator_loss", "Blstm_loss"]
            # labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, loc=7)
            plt.savefig(save_dir + "/figure.png")
            plt.savefig(save_dir + "/figure.pdf", bbox_inches='tight', pad_inches=0.01)
            # plt.show()
        else:
            # ax2.set_ylabel('Critic_loss & Blstm_loss ')
            # lns4 = ax2.plot(np.arange(total_batch), fig_b_loss_trains, 'tab:orange', label="Blstm_loss")
            # lns = lns1 + lns2 + lns4
            # labels = ["w_distance", "Critic_loss", "Blstm_loss"]
            pass


        ##############################w_distance ###############################
        if not is_pretrain:
            plt.figure()
            plt.plot(fig_time, fig_w_distance)
            plt.xlabel('Wallclock time (in seconds)')
            plt.ylabel('w_distance')
            plt.savefig(save_dir + '/figure_time_w_distance.png')
            # plt.show()
        ##############################Critic_loss ###############################
        if not is_pretrain:
            plt.figure()
            plt.plot(fig_time, fig_d_loss_trains)
            plt.xlabel('Wallclock time (in seconds)')
            plt.ylabel('Critic_loss')
            plt.savefig(save_dir + '/figure_time_Critic_loss.png')
            # plt.show()

        ##############################Generator_loss ###############################
        if not is_pretrain:
            plt.figure()
            plt.plot(fig_time, fig_g_loss_trains)
            plt.xlabel('Wallclock time (in seconds)')
            plt.ylabel('Generator_loss')
            plt.savefig(save_dir + '/figure_time_Generator_loss.png')
            # plt.show()
        ##############################BLSTM_loss ###############################
        plt.figure()
        plt.plot(fig_time, fig_b_loss_trains)
        plt.xlabel('Wallclock time (in seconds)')
        plt.ylabel('BLSTM_loss')
        plt.savefig(save_dir + '/figure_time_BLSTM_loss.png')
        # plt.show()

        # save data
        np.save(save_dir + '/fig_d_loss_trains.npy', np.array(fig_d_loss_trains))
        if not is_pretrain:
            np.save(save_dir + '/fig_g_loss_trains.npy', np.array(fig_g_loss_trains))

        np.save(save_dir + '/fig_b_loss_trains.npy', np.array(fig_b_loss_trains))
        np.save(save_dir + '/fig_w_distance.npy', np.array(fig_w_distance))
        np.save(save_dir + '/fig_time.npy', np.array(fig_time))
        plt.close()

    def record_time_info(self, folder):
        pass

    def eval(self, z):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.seq_size, self.z_dim)

    def save(self, checkpoint_dir, step):  # step代表参数更新了多少次
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        tf.train.Saver().save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            tf.train.Saver(self.blstm_params).restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
   ############替换#############################
    def save_final_info(self, folder, file_name, content):
        write_folder = self.outputdir + os.path.sep + folder
        if not os.path.exists(write_folder):
            os.mkdir(write_folder)
        result_file_path = write_folder + os.path.sep + file_name
        with open(result_file_path, "w+") as result_file:
            result_file.write(content)

    def save_model(self, e, folder, step, train_start_time, batch, fig_w_distance, fig_d_loss_trains, fig_g_loss_trains,
                   fig_b_loss_trains,
                   fig_time, is_pretrain=False):
        # saver.save(self.sess, self.checkpoint_dir+"/epoch_"+str(e+1))
        if not is_pretrain:
            data_to_write = []
            for i in range(160):
                z = make_noise([self.batch_size, self.z_dim])
                gen_samples = self.sess.run(self.fake_inputs_discrete_test, feed_dict={self.z: z})
                data_to_write.append(gen_samples)
            save_gen_samples(data_to_write, self.i2w, folder, self.outputdir)  # 保存生成器生成的报文
        ## 保存时间####
        write_to_content = "current_epoch:" + str(e + 1) + "\n"
        write_to_content += "current_step:" + str(step) + "\n"
        write_to_content += "After training, total time:" + str(
            timedelta(seconds=time.time() - train_start_time)) + "\n"

        self.save_final_info(folder, "result.txt", write_to_content)
        ## 保存model e+1 当前的epoch ###
        self.save(self.outputdir + os.path.sep + folder, step)
        ##画图,保存图片信息######
        fig_w_distance_t = fig_w_distance[:batch]
        fig_d_loss_trains_t = fig_d_loss_trains[:batch]
        fig_g_loss_trains_t = fig_g_loss_trains[:batch]
        fig_b_loss_trains_t = fig_b_loss_trains[:batch]
        fig_time_t = fig_time[:batch]

        self.draw_result_picture(batch, fig_w_distance_t, fig_d_loss_trains_t, fig_g_loss_trains_t, fig_b_loss_trains_t,
                                 folder, fig_time_t, is_pretrain)
