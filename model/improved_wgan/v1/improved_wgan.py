from datetime import timedelta
import sys
sys.path.append(r"/home/shenyuwang/paper_fuzzing/workspace") # model/improved_wgan/v1 
# print(sys.path)
import tensorflow as tf
from utils import *
import numpy as np
import time
from ops import *
import os
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import matplotlib
#
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow.contrib as tc

he_init = tf.contrib.layers.variance_scaling_initializer()
xavier_init = tf.contrib.layers.xavier_initializer()


class SA_GAN_SEQ(object):
    def __init__(self, sess, args, w2i, i2w):

        self.model_name = "SAGAN_SEQ"
        self.sess = sess

        self.w2i = w2i
        self.i2w = i2w
        self.outputdir = args.outputdir
        if not os.path.exists(self.outputdir):  # os模块判断并创建
            os.mkdir(self.outputdir)

        self.LAMBDA = args.LAMBDA  # 梯度惩罚
        self.epoch = args.epoch  # 一个epoch将所有的训练数据样本训练一次
        self.critic_iters = args.critic_iters  # 训练，鉴别器需要迭代的次数
        self.batch_size = args.batch_size  #
        self.seq_size = args.seq_size  # 报文x的长度
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.num_blocks = args.num_blocks
        self.d_ff = args.d_ff  # 前馈层参数

        self.z_dim = args.z_dim  # 噪声data的维数 50

        self.vocab_size = args.vocab_size

        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.g_beta1 = args.g_beta1
        self.g_beta2 = args.g_beta2

        self.d_beta1 = args.d_beta1
        self.d_beta2 = args.d_beta2

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
        self.fake_inputs = self.generator(self.z)  # shape (batch_size, seq_size, vocab_size), there is probablity
        self.fake_inputs_discrete = tf.argmax(self.fake_inputs, self.fake_inputs.get_shape().ndims - 1)

        self.fake_inputs_test = self.generator(self.z, False)
        self.fake_inputs_discrete_test = tf.argmax(self.fake_inputs_test, self.fake_inputs_test.get_shape().ndims - 1)

        self.real_logits = self.discriminator(self.real_inputs)
        self.fake_logits = self.discriminator(self.fake_inputs)

        self.disc_cost = tf.reduce_mean(self.fake_logits) - tf.reduce_mean(self.real_logits)
        self.gen_cost = -tf.reduce_mean(self.fake_logits)

        self.w_distance = tf.reduce_mean(self.real_logits) - tf.reduce_mean(self.fake_logits)

        self.gen_prob = tf.reduce_mean(tf.sigmoid(self.fake_logits))
        self.real_prob = tf.reduce_mean(tf.sigmoid(self.real_logits))

        # improved WGAN lipschitz-penalty
        self.alpha = tf.random_uniform(
            shape=[self.batch_size, 1, 1],
            minval=0.,
            maxval=1.
        )
        self.differences = self.fake_inputs - self.real_inputs  # batch_size seq_size vocab_size
        self.interpolates = self.real_inputs + (self.alpha * self.differences)
        self.gradients = tf.gradients(self.discriminator(self.interpolates), [self.interpolates])[
            0]  # 鉴别器函数对于 input self.interpolates 的导数
        self.slopes = tf.sqrt(
            tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2]))  # 将第二三维对应位置的梯度相加，每一个seq都有一个标量 梯度值
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)

        self.disc_cost += self.LAMBDA * self.gradient_penalty

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-4),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.gen_cost + self.reg
        self.d_loss_reg = self.disc_cost + self.reg
        # self.g_loss_reg = self.gen_cost
        # self.d_loss_reg = self.disc_cost

        self.variable = tf.trainable_variables()
        self.gen_params = [v for v in self.variable if 'Generator' in v.op.name]
        self.disc_params = [v for v in self.variable if 'discriminator' in v.op.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=self.g_beta1,
                                                       beta2=self.g_beta2).minimize(self.g_loss_reg,
                                                                                    var_list=self.gen_params,
                                                                                    colocate_gradients_with_ops=True)
            self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=self.d_beta1,
                                                        beta2=self.d_beta2).minimize(self.d_loss_reg,
                                                                                     var_list=self.disc_params,
                                                                                     colocate_gradients_with_ops=True)
            # self.disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4) \
            #     .minimize(self.d_loss_reg, var_list=self.disc_params)
            # self.gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4) \
            #     .minimize(self.g_loss_reg, var_list=self.gen_params)

        # weight clipping
        # self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.disc_params]

        self.d_sum = tf.summary.scalar("gen_cost", self.gen_cost)
        self.g_sum = tf.summary.scalar("disc_cost", self.disc_cost)

    def self_attention(self, input, input_mask=None, num_heads=4, is_training=True, scope="multihead_attention"):
        return multihead_attention(queries=input, # 每一个head的dim必须是偶数，否则添加相对位置信息的时候会报错。dim//head
                                   keys=input,
                                   values=input,
                                   key_masks=input_mask,
                                   query_masks=input_mask,
                                   num_heads=num_heads,
                                   dropout_rate=self.dropout_rate,
                                   training=is_training,
                                   kernel_initializer=xavier_init,
                                   causality=False,
                                   scope=scope
                                   )

    def generator(self, z, is_training=True):
        """
        achitecture:

        [one] linear layer
        [num_blocks] transformer block
        [one] linear layer
        [one] softmax layer
        """
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):  # z's shape (batch_size, z_dim)
            x = tf.layers.dense(z, units=self.seq_size * self.d_model,
                                kernel_initializer=he_init)  # batch_size, seq_size
            x = tf.nn.selu(x)
            x = tf.reshape(x, [self.batch_size, self.seq_size, self.d_model])
            x *= self.d_model ** 0.5 # 进行缩放
            # x += positional_encoding(x, self.seq_size)
            # x = tf.layers.dropout(x, self.dropout_rate, training=is_training)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    x = self.self_attention(x, num_heads=self.num_heads, is_training=is_training)
                    # x = multihead_attention(queries=x,
                    #                         keys=x,
                    #                         values=x,
                    #                         num_heads=self.num_heads,
                    #                         dropout_rate=self.dropout_rate,
                    #                         training=is_training,
                    #                         kernel_initializer=xavier_init,
                    #                         causality=False)
                    x = ff(x, num_units=[self.d_ff, self.d_model])
            # x = tf.transpose(x, [0, 2, 1])   # before:(batch_size, d_model , seq_len)  ---->   after transpose: (batch_size, d_model , seq_len)
            # x = self.ResBlock('ResBlock', x)
            # x = tf.transpose(x, [0, 2, 1])

            weights = tf.get_variable(dtype=tf.float32, shape=(self.d_model, self.vocab_size), initializer=xavier_init,
                                      name="weights")
            logits = tf.einsum('ntd,dk->ntk', x, weights)  # (batch_size, seq_size, vocab_size)
            # res = tf.reshape(tf.argmax(logits, axis=2), [self.batch_size, self.seq_size])
        return tf.nn.softmax(logits)

    def discriminator(self, x, is_training=True):
        # input_ids = tf.argmax(x, x.get_shape().ndims - 1, output_type=tf.int32)
        #
        # input_mask_tmp = tf.where(tf.equal(16, input_ids), 0 * tf.ones_like(input_ids), input_ids)
        # input_mask = tf.where(tf.logical_not(tf.equal(16, input_mask_tmp)), 1 * tf.ones_like(input_mask_tmp),
        #                       input_mask_tmp)
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            # 原版
            # 新版
            output = tf.transpose(x, [0, 2, 1])
            output = lib.ops.conv1d.Conv1D('Conv1d.1', self.vocab_size, self.d_model, 1,
                                           output)  # (batch_size, d_model, seq_size)
            output = tf.transpose(output, [0, 2, 1])  # (batch_size, seq_size, d_model)
            # output = output + positional_encoding(output, self.seq_size)
            # output = self.self_attention(output,input_mask, is_training=is_training,scope="multihead_attention_1")  # (batch_size, seq_size, vocab_size)

            output = tf.transpose(output, [0, 2, 1])
            output = lib.ops.conv1d.Conv1D('Conv1d.2', self.d_model, self.d_model, 2, output)
            output = tf.transpose(output, [0, 2, 1])  # (batch_size, seq_size, d_model)
            # output = self.self_attention(output, is_training=is_training)

            output = tf.transpose(output, [0, 2, 1])
            output = lib.ops.conv1d.Conv1D('Conv1d.3', self.d_model, self.d_model, 3, output)
            output = tf.transpose(output, [0, 2, 1])  # (batch_size, seq_size, d_model)
            # output = self.self_attention(output, is_training=is_training)

            output = tf.transpose(output, [0, 2, 1])
            output = lib.ops.conv1d.Conv1D('Conv1d.4', self.d_model, self.d_model, 4, output)
            output = tf.transpose(output, [0, 2, 1])  # (batch_size, seq_size, d_model)
            # output = self.self_attention(output, is_training=is_training)

            output = tf.transpose(output, [0, 2, 1])
            output = lib.ops.conv1d.Conv1D('Conv1d.5', self.d_model, self.d_model, 5, output)
            output = tf.transpose(output, [0, 2, 1])  # (batch_size, seq_size, d_model)
            output1 = self.self_attention(output, is_training=is_training, scope="multihead_attention_1")
            output2 = self.self_attention(output1, is_training=is_training, scope="multihead_attention_2")
            output = output1 + output2
            res = tf.reshape(output, [self.batch_size, self.d_model * self.seq_size])
            res = tf.layers.dense(res, units=1, kernel_initializer=xavier_init)

        return res

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def ResBlock(self, name, inputs):
        output = inputs
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name + '.1', self.d_model, self.d_model, 5, output)
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name + '.2', self.d_model, self.d_model, 5, output)
        return inputs + (0.3 * output)

    def train(self, data):
        batch = 0
        step = 0  # 代表参数的更新次数
        min_w_distance = 1e10
        epoch_need_record = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        n_batch = len(data) // self.batch_size  # 总batch数
        total_batch = n_batch // 100 * self.epoch  ## 每隔100个batch 记录一次，因此一共记录（n_batch // 100） * self_epoch次
        print("n_batch:", n_batch)
        print("total_batch", n_batch // 100 * self.epoch)

        # saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # 画图相关
        fig_w_distance = np.zeros([total_batch])
        fig_d_loss_trains = np.zeros([total_batch])
        fig_g_loss_trains = np.zeros([total_batch])
        fig_time = np.zeros([total_batch])

        train_start_time = time.time()
        for e in range(self.epoch):
            real_data = inf_train_gen(data, self.batch_size)
            epoch_start_time = time.time()
            iter = 0
            epoch_over = False
            while not epoch_over:
                z = make_noise(shape=[self.batch_size, self.z_dim])
                self.sess.run(self.gen_train_op, feed_dict={self.z: z})
                for _ in range(self.critic_iters):
                    iter += 1
                    step += 1
                    real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                    z = make_noise([self.batch_size, self.z_dim])
                    # self.sess.run(self.d_clip)
                    _disc_cost, _gen_cost, _ = self.sess.run([self.disc_cost, self.gen_cost, self.disc_train_op],
                                                             feed_dict={self.real_inputs_discrete: real_inputs_discrete,
                                                                        self.z: z})
                    if iter % 100 == 99:
                        z = make_noise([self.batch_size, self.z_dim])
                        iter += 1
                        real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                        gen_samples, gen_prob, real_prob, disc_cost, gen_cost, w_distance = \
                            self.sess.run([self.fake_inputs_discrete, self.gen_prob, self.real_prob, self.disc_cost,
                                           self.gen_cost, self.w_distance],
                                          feed_dict={self.real_inputs_discrete: real_inputs_discrete, self.z: z})
                        fig_w_distance[batch] = w_distance
                        fig_d_loss_trains[batch] = disc_cost
                        fig_g_loss_trains[batch] = gen_cost
                        fig_time[batch] = time.time() - train_start_time
                        batch += 1

                        translate(gen_samples, self.i2w)
                        time.sleep(0.1)
                        print(
                            "Epoch {}\niter {}\ndisc cost {}, real prob {}\ngen cost {}, gen prob {}\nw-distance {}\ntime{}"
                                .format(e + 1, iter, disc_cost, real_prob, gen_cost, gen_prob, w_distance,
                                        timedelta(seconds=time.time() - epoch_start_time)))
                        if min_w_distance > w_distance:
                            min_w_distance = w_distance
                            self.save_model(e, "best_model", step, train_start_time, batch, fig_w_distance,
                                            fig_d_loss_trains, fig_g_loss_trains, fig_time)
                    # print("iter:", iter)

                    if iter % n_batch == 0:  ## 每一次训练集中的数据训练完一遍的时候

                        # 某些epoch存checkpoint并且保存生成的数据
                        if (e + 1) % 10 == 0:
                            self.save_model(e, 'epoch' + str(e + 1), step, train_start_time, batch, fig_w_distance,
                                            fig_d_loss_trains, fig_g_loss_trains, fig_time)
                        epoch_over = True
                        break

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

        ###########################   绘图 start   #######################################
        # 绘制曲线
        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # lns1 = ax1.plot(np.arange(total_batch), fig_w_distance, label="w_distance")
        # # 按一定间隔显示实现方法
        # # ax2.plot(200 * np.arange(len(fig_accuracy)), fig_accuracy, 'r')
        # lns2 = ax2.plot(np.arange(total_batch), fig_d_loss_trains, 'r', label="Critic_loss")
        # lns3 = ax2.plot(np.arange(total_batch), fig_g_loss_trains, 'g', label="Generator_loss")
        # ax1.set_xlabel('generator Iterations')
        # ax1.set_ylabel('w_distance')
        # ax2.set_ylabel('Critic_loss & Generator_loss')
        #
        # # 合并图例
        # lns = lns1 + lns2 + lns3
        # labels = ["w_distance", "Critic_loss", "Generator_loss"]
        # # labels = [l.get_label() for l in lns]
        # plt.legend(lns, labels, loc=7)
        # plt.savefig(self.outputdir + "/figure.png")
        # plt.savefig(self.outputdir + "/figure.pdf", bbox_inches='tight', pad_inches=0.01)
        # plt.show()
        #
        # plt.figure()
        # plt.plot(fig_time, fig_w_distance)
        # plt.xlabel('Wallclock time (in seconds)')
        # plt.ylabel('w_distance')
        # plt.savefig(self.outputdir + '/figure_time.png')
        # plt.show()
        #
        # # save data
        # np.save(self.outputdir + '/fig_d_loss_trains.npy', np.array(fig_d_loss_trains))
        # np.save(self.outputdir + '/fig_g_loss_trains.npy', np.array(fig_g_loss_trains))
        # np.save(self.outputdir + '/fig_w_distance.npy', np.array(fig_w_distance))
        # np.save(self.outputdir + '/fig_time.npy', np.array(fig_time))
        # print("Saved d_loss & g_loss & w_distance & trainning time list ")
        # # np.load(self.outputdir+'/fig_d_loss_trains.npy')  # load contexts from store file
        ###########################   绘图 end   #######################################
        lib._params.clear()

    def draw_result_picture(self, total_batch, fig_w_distance, fig_d_loss_trains, fig_g_loss_trains, folder,
                            fig_time):
        save_dir = self.outputdir + os.path.sep + folder
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(np.arange(total_batch), fig_w_distance, label="w_distance")
        # 按一定间隔显示实现方法
        # ax2.plot(200 * np.arange(len(fig_accuracy)), fig_accuracy, 'r')
        lns2 = ax2.plot(np.arange(total_batch), fig_d_loss_trains, 'r', label="Critic_loss")
        lns3 = ax2.plot(np.arange(total_batch), fig_g_loss_trains, 'g', label="Generator_loss")
        ax1.set_xlabel('generator Iterations')
        ax1.set_ylabel('w_distance')
        ax2.set_ylabel('Critic_loss & Generator_loss')

        # 合并图例
        lns = lns1 + lns2 + lns3
        labels = ["w_distance", "Critic_loss", "Generator_loss"]
        # labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc=7)
        plt.savefig(save_dir + "/figure.png")
        plt.savefig(save_dir + "/figure.pdf", bbox_inches='tight', pad_inches=0.01)
        # plt.show()

        plt.figure()
        plt.plot(fig_time, fig_w_distance)
        plt.xlabel('Wallclock time (in seconds)')
        plt.ylabel('w_distance')
        plt.savefig(save_dir + '/figure_time.png')
        # plt.show()

        # save data
        np.save(save_dir + '/fig_d_loss_trains.npy', np.array(fig_d_loss_trains))
        np.save(save_dir + '/fig_g_loss_trains.npy', np.array(fig_g_loss_trains))
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
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save_final_info(self, folder, file_name, content):
        write_folder = self.outputdir + os.path.sep + folder
        if not os.path.exists(write_folder):
            os.mkdir(write_folder)
        result_file_path = write_folder + os.path.sep + file_name
        with open(result_file_path, "w+") as result_file:
            result_file.write(content)

    def save_model(self, e, folder, step, train_start_time, batch, fig_w_distance, fig_d_loss_trains, fig_g_loss_trains,
                   fig_time):
        # saver.save(self.sess, self.checkpoint_dir+"/epoch_"+str(e+1))
        data_to_write = []
        for i in range(160):
            z = make_noise([self.batch_size, self.z_dim])
            gen_samples = self.sess.run(self.fake_inputs_discrete, feed_dict={self.z: z})
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
        fig_time_t = fig_time[:batch]
        self.draw_result_picture(batch, fig_w_distance_t, fig_d_loss_trains_t, fig_g_loss_trains_t,
                                 folder, fig_time_t)
