# coding=utf-8
# 鸡鸣万户晓 鹤舞一年春
# 戌岁祝福万事顺 狗年兆丰五谷香
from datetime import timedelta
import os
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils import *
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Seq_Gan_SEQ(object):
    def __init__(self, sess, args, w2i, i2w):
        self.model_name = "LSTM_SEQ"
        self.outputdir = args.outputdir
        if not os.path.exists(self.outputdir):  # os模块判断并创建
            os.mkdir(self.outputdir)
        # 超参数
        self.encoding_embedding_size = args.encoding_embedding_size
        self.decoding_embedding_size = args.decoding_embedding_size

        self.source_vocab_size = args.vocab_size
        self.target_vocab_size = args.vocab_size
        self.num_layers = args.num_layers
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.rnn_size = args.rnn_size
        self.learning_rate = args.lr
        self.w2i = w2i
        self.i2w = i2w
        self.sess = sess

        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')
        self.source_data = tf.placeholder(tf.int32, shape=[self.batch_size, None],name="source_data")
        self.targets_data = tf.placeholder(tf.int32, shape=[self.batch_size, None],name="targets_data")



        self.training_decoder_output, self.predicting_decoder_output = self.seq2seq_model(self.source_data,
                                                                                          self.targets_data)
        self.training_logits = tf.identity(self.training_decoder_output.rnn_output, 'logits')
        self.predicting_logits = tf.identity(self.predicting_decoder_output.sample_id, name='predictions')
        #########计算损失######################
        self.masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32,
                                      name="masks")
        self.cost = tf.contrib.seq2seq.sequence_loss(
            self.training_logits,
            self.targets_data,
            self.masks
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.gradients = self.optimizer.compute_gradients(self.cost)
        self.capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in self.gradients if
                                 grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.capped_gradients)

    ###################################################################################


    # Encoder
    """
    在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
    在Embedding中，我们使用tf.contrib.layers.embed_sequence，它会对每个batch执行embedding操作。
    """

    def get_encoder_layer(self, input_data):
        """
        构造Encoder层
        参数说明：
        - input_data: 输入tensor
        - rnn_size: rnn隐层结点数量
        - num_layers: 堆叠的rnn cell数量
        - source_sequence_length: 源数据的序列长度
        - source_vocab_size: 源数据的词典大小
        - encoding_embedding_size: embedding的大小
        """
        # https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/layers/embed_sequence
        """
        embed_sequence(
        ids,
        vocab_size=None,
        embed_dim=None,
        unique=False,
        initializer=None,
        regularizer=None,
        trainable=True,
        scope=None,
        reuse=None
        )
        ids: [batch_size, doc_length] Tensor of type int32 or int64 with symbol ids.

        return : Tensor of [batch_size, doc_length, embed_dim] with embedded sequences.
        """
        encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, self.source_vocab_size,
                                                               self.encoding_embedding_size)

        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])

        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=self.source_sequence_length, dtype=tf.float32)

        return encoder_output, encoder_state

    def process_decoder_input(self, data):

        ending = tf.strided_slice(data, [0, 0], [self.batch_size, -1],
                                  [1, 1])  # 相当于是取batch_size条数据 [batch_size,seq_len]
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.w2i['s']), ending], 1)  # 在每一个seq前面加上<GO>字符，代表起始字符

        return decoder_input

    def decoding_layer(self, encoder_state, decoder_input):
        '''
        构造Decoder层
        参数：
        - target_letter_to_int: target数据的映射表
        - decoding_embedding_size: embed向量大小
        - num_layers: 堆叠的RNN单元数量
        - rnn_size: RNN单元的隐层结点数量
        - target_sequence_length: target数据序列长度
        - max_target_sequence_length: target数据序列最大长度
        - encoder_state: encoder端编码的状态向量
        - decoder_input: decoder端输入
        '''

        # 1. Embedding
        decoder_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

        # 构造Decoder中的RNN单元
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                                   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return decoder_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(self.rnn_size) for _ in range(self.num_layers)])

        # Output全连接层
        # target_vocab_size定义了输出层的大小
        output_layer = Dense(self.target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

        # 4. Training decoder
        with tf.variable_scope("decode"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=self.target_sequence_length,
                                                                time_major=False)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                              maximum_iterations=self.max_target_sequence_length)

        # 5. Predicting decoder
        # 与training共享参数

        with tf.variable_scope("decode", reuse=True):
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([self.w2i['s']], dtype=tf.int32), [self.batch_size],
                                   name='start_token')
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,
                                                                         self.w2i['n'])

            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                                 predicting_helper,
                                                                 encoder_state,
                                                                 output_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                impute_finished=True,
                                                                                maximum_iterations=self.max_target_sequence_length)

        return training_decoder_output, predicting_decoder_output

    # 上面已经构建完成Encoder和Decoder，下面将这两部分连接起来，构建seq2seq模型
    def seq2seq_model(self, source_data, targets_data):

        _, encoder_state = self.get_encoder_layer(source_data)

        decoder_input = self.process_decoder_input(targets_data)

        training_decoder_output, predicting_decoder_output = self.decoding_layer(encoder_state, decoder_input)

        return training_decoder_output, predicting_decoder_output

    # Train
    def train(self, data):
        batch = 0
        step = 0
        n_batch = len(data) // self.batch_size  # 总batch数

        # self.sess.run(tf.global_variables_initializer())

        train_start_time = time.time()
        total_batch = n_batch // 100 * self.epochs  ## 每隔100个batch 记录一次，因此一共记录（n_batch // 100） * self_epoch次
        fig_loss_trains = np.zeros([total_batch])
        fig_time = np.zeros([total_batch])

        #################################################预训练#######################################################
        real_data = inf_train_gen(data, self.batch_size,self.w2i)
        for i in range(5):
            sources_batch, targets_batch, sources_lengths, targets_lengths = next(real_data)

            training_logits, predicting_logits = self.sess.run([self.training_logits, self.predicting_logits], feed_dict={
                self.source_data: sources_batch,
                self.targets_data: targets_batch,
                self.target_sequence_length: targets_lengths,
                self.source_sequence_length: sources_lengths
            })
            # print(training_logits)
            # print(predicting_logits)

            # _, loss = self.sess.run([self.train_op, self.cost], feed_dict={
            #     self.source_data: sources_batch,
            #     self.targets_data: targets_batch,
            #     self.target_sequence_length: targets_lengths,
            #     self.source_sequence_length: sources_lengths
            # })

        ##################################################正式训练####################################################
        for epoch_i in range(self.epochs):
            iter = 0
            real_data = inf_train_gen(data, self.batch_size,self.w2i)
            epoch_start_time = time.time()
            for i in range(n_batch):
                sources_batch, targets_batch, sources_lengths, targets_lengths = next(real_data)
                _, loss = self.sess.run([self.train_op, self.cost], feed_dict={
                    self.source_data: sources_batch,
                    self.targets_data: targets_batch,
                    self.target_sequence_length: targets_lengths,
                    self.source_sequence_length: sources_lengths
                })
                step = step + 1

                if (iter+1) % 100 == 0:
                    fig_time[batch] = time.time() - train_start_time
                    fig_loss_trains[batch] = loss
                    batch = batch+1
                    print(
                        "Epoch {}\niter {}\nloss {} \ntime{}\n"
                            .format(epoch_i + 1, iter+1, loss,
                                    timedelta(seconds=time.time() - epoch_start_time)))
                iter += 1


            if (epoch_i+1) % 10 == 0:
                self.save_model(data, epoch_i, 'epoch' + str(epoch_i + 1), step, train_start_time, batch, fig_loss_trains, fig_time)

                # if batch_i % display_step == 0:
                #     # 计算validation loss
                #     validation_loss = sess.run(
                #         [cost],
                #         {input_data: valid_sources_batch,
                #          targets: valid_targets_batch,
                #          lr: learning_rate,
                #          target_sequence_length: valid_targets_lengths,
                #          source_sequence_length: valid_sources_lengths})
                #
                #     print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                #           .format(epoch_i,
                #                   epochs,
                #                   batch_i,
                #                   len(train_source) // batch_size,
                #                   loss,
                #                   validation_loss[0]))



    # # 预测
    def generate_data(self,data,cur_epoch,train_end_time):
        # data_to_write = []
        total_frame_num = 0
        real_data = inf_train_gen(data, self.batch_size, self.w2i)
        n_batch = len(data) // self.batch_size  # 总batch数
        cur_batch = 0
        while True:
            cur_batch = cur_batch+1
            if cur_batch > n_batch:
                real_data = inf_train_gen(data, self.batch_size, self.w2i)
                  
            sources_batch, targets_batch, sources_lengths, targets_lengths = next(real_data)
            gen_samples = self.sess.run(self.predicting_logits, feed_dict={
                self.source_data: sources_batch,
                self.target_sequence_length:  targets_lengths,
                self.source_sequence_length: sources_lengths
            })
            total_frame_num = total_frame_num + self.batch_size

            if (time.time()-train_end_time) >= 0:
                break
            # data_to_write.append(gen_samples)
        # save_gen_samples(data_to_write, self.i2w, cur_epoch, self.outputdir)  # 保存生成器生成的报文
        with open(self.outputdir+os.path.sep+"gen_result.txt","w") as wf:
            wf.write("total_frame_num:"+str(total_frame_num)+"\n")
            wf.write("exceed_seconds:"+str(time.time()-train_end_time)+"\n")


    # def source_to_seq(self,text):
    #     sequence_length = 7
    #     return [self.w2i.get(word, self.w2i['<UNK>']) for word in text] + [self.w2i['u']] * (sequence_length - len(text))

    def draw_result_picture(self, total_batch, fig_loss_trains, folder, fig_time):
        save_dir = self.outputdir + os.path.sep + folder
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        fig, ax1 = plt.subplots()

        lns1 = ax1.plot(np.arange(total_batch), fig_loss_trains, 'g', label="Generator_loss")
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('loss')

        lns = lns1
        labels = ["lstm_loss"]
        plt.legend(lns, labels, loc=7)
        plt.savefig(save_dir + "/figure.png")
        plt.savefig(save_dir + "/figure.pdf", bbox_inches='tight', pad_inches=0.01)
        # plt.show()

        ##############################LSTM_loss ###############################
        plt.figure()
        plt.plot(fig_time, fig_loss_trains)
        plt.xlabel('Wallclock time (in seconds)')
        plt.ylabel('LSTM_loss')
        plt.savefig(save_dir + '/figure_time_LSTM_loss.png')
        # plt.show()

        # save data
        np.save(save_dir + '/LSTM_loss.npy', np.array(fig_loss_trains))
        np.save(save_dir + '/fig_time.npy', np.array(fig_time))
        plt.close()

    @property
    def model_dir(self):
        return "{}".format(self.model_name)

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

    ############替换#############################
    def save_final_info(self, folder, file_name, content):
        write_folder = self.outputdir + os.path.sep + folder
        if not os.path.exists(write_folder):
            os.mkdir(write_folder)
        result_file_path = write_folder + os.path.sep + file_name
        with open(result_file_path, "w+") as result_file:
            result_file.write(content)

    def save_model(self, data, e, folder, step, train_start_time, batch, fig_loss_trains, fig_time):

        ###################################生成样本#############################
        data_to_write = []
        real_data = inf_train_gen(data, self.batch_size, self.w2i)
        n_batch = len(data) // self.batch_size  # 总batch数
        cur_batch = 0
        for i in range(160):
            cur_batch = cur_batch+1
            if cur_batch > n_batch:
                real_data = inf_train_gen(data, self.batch_size, self.w2i)
                  
            sources_batch, targets_batch, sources_lengths, targets_lengths = next(real_data)
            gen_samples = self.sess.run(self.predicting_logits, feed_dict={
                self.source_data: sources_batch,
                self.target_sequence_length:  targets_lengths,
                self.source_sequence_length: sources_lengths
            })
            data_to_write.append(gen_samples)
        save_gen_samples(data_to_write, self.i2w, folder, self.outputdir)  # 保存生成器生成的报文
        ## 保存时间##############################################
        write_to_content = "current_epoch:" + str(e + 1) + "\n"
        write_to_content += "current_step:" + str(step) + "\n"
        write_to_content += "After training, total time:" + str(
            timedelta(seconds=time.time() - train_start_time)) + "\n"

        self.save_final_info(folder, "result.txt", write_to_content)
        ## 保存model e+1 当前的epoch ###
        self.save(self.outputdir + os.path.sep + folder, step)
        ##画图,保存图片信息######

        fig_loss_trains = fig_loss_trains[:batch]

        fig_time_t = fig_time[:batch]

        self.draw_result_picture(batch, fig_loss_trains, folder, fig_time_t)
