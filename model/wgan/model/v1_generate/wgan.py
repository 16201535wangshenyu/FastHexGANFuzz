import math
from datetime import timedelta

import tensorflow as tf
import time
from utils import *
import os
# import tflib as lib
# import tflib.ops.linear
# import tflib.ops.batchnorm
# import tflib.ops.conv1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

he_init = tf.contrib.layers.variance_scaling_initializer()
xavier_init = tf.contrib.layers.xavier_initializer()


class WGAN_SEQ(object):
    def __init__(self, sess, args, w2i, i2w,real_data_loader):
        self.real_data_loader = real_data_loader
        self.model_name = "WGAN_SEQ"
        self.sess = sess

        self.w2i = w2i
        self.i2w = i2w
        self.outputdir = args.outputdir
        if not os.path.exists(self.outputdir):  # os模块判断并创建
            os.mkdir(self.outputdir)

        self.epoch = args.epoch  # 一个epoch将所有的训练数据样本训练一次
        self.critic_iters = args.critic_iters  # 训练，鉴别器需要迭代的次数
        self.batch_size = args.batch_size  #
        self.seq_size = args.seq_size  # 报文x的长度


        self.num_blocks = args.num_blocks


        self.z_dim = args.z_dim  # 噪声data的维数 50

        self.vocab_size = args.vocab_size

        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr

        self.d_model = args.d_model  # # 前馈层参数

        #################################

        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])

        self.fake_inputs = self.generator(self.z) # [batch_size,seq_size]
        self.gen_samples = tf.round(tf.multiply(self.fake_inputs,self.vocab_size-1)) # [batch_size,seq_size]
        self.real_inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_size])  # [batch_size,seq_size]

        self.real_logits = self.discriminator(self.real_inputs)
        self.fake_logits = self.discriminator(self.fake_inputs)

        self.disc_cost = tf.reduce_mean(self.fake_logits) - tf.reduce_mean(self.real_logits)
        self.gen_cost = -tf.reduce_mean(self.fake_logits)

        self.w_distance = tf.reduce_mean(self.real_logits) - tf.reduce_mean(self.fake_logits)

        self.gen_prob = tf.reduce_mean(tf.sigmoid(self.fake_logits))
        self.real_prob = tf.reduce_mean(tf.sigmoid(self.real_logits))

        # improved WGAN lipschitz-penalty

        # self.reg = tc.layers.apply_regularization(
        #     tc.layers.l1_regularizer(2.5e-4),
        #     weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        # )
        # self.g_loss_reg = self.gen_cost + self.reg
        # self.d_loss_reg = self.disc_cost + self.reg
        self.g_loss_reg = self.gen_cost
        self.d_loss_reg = self.disc_cost

        self.variable = tf.trainable_variables()
        self.gen_params = [v for v in self.variable if 'Generator' in v.op.name]
        self.disc_params = [v for v in self.variable if 'discriminator' in v.op.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=self.g_beta1,
            #                                            beta2=self.g_beta2).minimize(self.g_loss_reg,
            #                                                                         var_list=self.gen_params,
            #                                                                         colocate_gradients_with_ops=True)
            # self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=self.d_beta1,
            #                                             beta2=self.d_beta2).minimize(self.d_loss_reg,
            #                                                                          var_list=self.disc_params,
            #                                                                          colocate_gradients_with_ops=True)
            self.disc_train_op = tf.train.RMSPropOptimizer(learning_rate=self.d_learning_rate) \
                .minimize(self.d_loss_reg, var_list=self.disc_params)
            self.gen_train_op = tf.train.RMSPropOptimizer(learning_rate=self.g_learning_rate) \
                .minimize(self.g_loss_reg, var_list=self.gen_params)

        # weight clipping
        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.disc_params]

        self.d_sum = tf.summary.scalar("gen_cost", self.gen_cost)
        self.g_sum = tf.summary.scalar("disc_cost", self.disc_cost)

    def ReLULayer(self, name, n_in, n_out, inputs):
        output = tf.layers.dense(inputs,n_out,kernel_initializer=xavier_init,name=name + '.Linear')
        # output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
        return tf.nn.relu(output)

    def generator(self, z, is_training=True):
        """
        achitecture:

        [one] linear layer
        [num_blocks] transformer block
        [one] linear layer
        [one] softmax layer
        """
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):  # z's shape (batch_size, z_dim)
            # output = lib.ops.batchnorm.Batchnorm("Generator.InputNormalize", [0, 1], z, fused=True)
            output = self.ReLULayer('Generator.InputLinear', self.z_dim, self.d_model, z)
            for i in range(self.num_blocks):
                output = self.ReLULayer('Generator.hidden_layer_{}'.format(i), self.d_model, self.d_model, output)

            # output = lib.ops.linear.Linear("Generator.output", self.d_model, self.seq_size, output,initialization='he')
            output = tf.layers.dense(output, self.seq_size, kernel_initializer=xavier_init, name="Generator.output")
            output = tf.sigmoid(output) # [batch_size,seq_size]

        return output

    def discriminator(self, x, is_training=True):
        """
        # x:  (batch_size, seq_size)
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            # x = tf.reshape(x,[-1,self.seq_size * self.vocab_size])
            # output = lib.ops.batchnorm.Batchnorm("discriminator.InputNormalize", [0, 1], x, fused=True)
            output = self.ReLULayer('discriminator.InputLinear', self.seq_size, self.d_model, x)

            output = self.ReLULayer('discriminator.hidden_layer_1', self.d_model, self.d_model, output)

            # output = lib.ops.linear.Linear("discriminator.output", self.d_model, 1, output,initialization='he')
            output = tf.layers.dense(output, 1, kernel_initializer=xavier_init, name="discriminator.output")
        return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def train(self, data):
        batch = 0
        step = 0  # 代表参数的更新次数
        min_w_distance = 1e10

        n_batch = len(data) // self.batch_size  # 总batch数
        total_batch = n_batch // 100 * self.epoch  ## 每隔100个batch 记录一次，因此一共记录（n_batch // 100） * self_epoch次
        print("n_batch:", n_batch)
        print("total_batch", n_batch // 100 * self.epoch)

        # saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())

        # 画图相关
        fig_w_distance = np.zeros([total_batch])
        fig_d_loss_trains = np.zeros([total_batch])
        fig_g_loss_trains = np.zeros([total_batch])
        fig_time = np.zeros([total_batch])

        train_start_time = time.time()
        for e in range(self.epoch):
            self.real_data_loader.reset_pointer()
            epoch_start_time = time.time()
            iter = 0
            epoch_over = False
            while not epoch_over:
                z = make_noise(shape=[self.batch_size, self.z_dim])
                self.sess.run(self.gen_train_op, feed_dict={self.z: z})
                for _ in range(self.critic_iters):
                    iter += 1
                    step += 1
                    real_inputs = self.real_data_loader.next_batch()  # (batch_size, seq_len)
                    z = make_noise([self.batch_size, self.z_dim])

                    self.sess.run(self.d_clip)
                    _disc_cost, _gen_cost, _ = self.sess.run([self.disc_cost, self.gen_cost, self.disc_train_op],
                                                             feed_dict={self.real_inputs: real_inputs,
                                                                        self.z: z})

                    if iter % 100 == 99:
                        z = make_noise([self.batch_size, self.z_dim])
                        iter += 1
                        real_inputs = self.real_data_loader.next_batch()  # (batch_size, seq_len)  # (batch_size, seq_len)
                        gen_samples, gen_prob, real_prob, disc_cost, gen_cost, w_distance = \
                            self.sess.run([self.gen_samples, self.gen_prob, self.real_prob, self.disc_cost,
                                           self.gen_cost, self.w_distance],
                                          feed_dict={self.real_inputs: real_inputs, self.z: z})
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
                        if min_w_distance > abs(w_distance):
                            min_w_distance = abs(w_distance)
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


        print("After training, total time:", timedelta(seconds=time.time() - train_start_time))
        print("w_sitance")
        print(fig_w_distance)
        print("\nd_loss")
        print(fig_d_loss_trains)
        print("\ng_loss")
        print(fig_g_loss_trains)
        print("\ntime")
        print(fig_time)

        # lib._params.clear()

    # # 预测
    def generate_data(self,cur_epoch,train_end_time):
        total_frame_num = 0
        # data_to_write = []
        while True:
            z = make_noise([self.batch_size, self.z_dim])
            gen_samples = self.sess.run(self.gen_samples, feed_dict={self.z: z})
            total_frame_num = total_frame_num + self.batch_size
            if (time.time()-train_end_time) >= 0:
                break
            # data_to_write.append(gen_samples)
        # save_gen_samples(data_to_write, self.i2w, cur_epoch, self.outputdir)  # 保存生成器生成的报文
        with open(self.outputdir+os.path.sep+"gen_result.txt","w") as wf:
            wf.write("total_frame_num:"+str(total_frame_num)+"\n")
            wf.write("exceed_seconds:"+str(time.time()-train_end_time)+"\n")

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
            tf.train.Saver().restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
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
            gen_samples = self.sess.run(self.gen_samples, feed_dict={self.z: z})
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