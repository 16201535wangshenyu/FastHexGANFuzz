import time
import sys
sys.path.append(r"/home/shenyuwang/paper_fuzzing/workspace/model/seqgan")
import tensorflow as tf
import os
import numpy as np

from configuration import training_config,generator_config,discriminator_config
from dataloader import Gen_Data_loader,Dis_dataloader
from generator import Generator
from rollout import rollout
from discriminator import Discriminator
import pickle
from utils import generate_samples,save_model,load,generate_real_sample
import traceback

# config_hardware = tf.ConfigProto()
# config_hardware.gpu_options.per_process_gpu_memory_fraction = 0.40
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
'''
返回一个目录下所有文件夹
返回格式{“文件名”：“文件绝对路径”}
'''

def retrun_allfiles_from_folder(folder):
    file_abs = {}
    if os.path.isfile(folder): # 如果是文件
        filename = os.path.basename(folder)
        file_abs[filename] = folder
        print("this is a file")
    if os.path.isdir(folder):
        files = os.listdir(folder)
        folder_abs = os.path.abspath(folder)
        for file in files:
            path = folder_abs + os.path.sep + file
            file_abs[file] = path
        print("this is a folder")
    return file_abs


def init_args_set(data_set_file_list, output_dir_root):
    times = time.time()
    local_time = time.localtime(times)
    args_set = []
    output_dir_root_abs = os.path.abspath(output_dir_root)
    for file in data_set_file_list:
        args = []
        file_abs = data_set_file_list[file]
        seq_size = int(file.split('_')[-2])
        output_dir_name = "test" + str(time.strftime("%Y%m%d_%H_%M_%S", local_time)) + "_" + file[0:-4]
        args.append(output_dir_root_abs + os.path.sep + output_dir_name)
        args.append(file_abs)
        args.append(seq_size)
        args_set.append(args)
    return args_set

def init_args_set_testing(data_set_file_list, output_dir_root):
    times = time.time()
    local_time = time.localtime(times)
    args_set = []
    output_dir_root_abs = os.path.abspath(output_dir_root)
    output_file_list = os.listdir(output_dir_root_abs)
    for file in data_set_file_list:
        args = []
        file_abs = data_set_file_list[file]
        seq_size = int(file.split('_')[-2])
        output_dir_name = ""
        for output_file in output_file_list:
            if file[0:-4] in output_file:
                output_dir_name = output_file
                break
        # output_dir_name = "test" + str(time.strftime("%Y%m%d_%H_%M_%S", local_time)) + "_" + file[0:-4]
        if output_dir_name != "":
            args.append(output_dir_root_abs + os.path.sep + output_dir_name)
            args.append(file_abs)
            args.append(seq_size)
            args_set.append(args)
    return args_set

def training():
    data_resp_list = [
        r"/home/shenyuwang/paper_fuzzing/workspace/data/coap/cluster" , ## 存放一个个数据集的文件夹
        # r"/home/shenyuwang/paper_fuzzing/workspace/data/modbus/trian_data",
        # r"/home/shenyuwang/paper_fuzzing/workspace/data/mqtt/cluster"  ## 存放一个个数据集的文件夹
    ]
    output_dir_root_list = [
        r"/home/shenyuwang/paper_fuzzing/workspace/output/coap/seqgan",
        # r"/home/shenyuwang/paper_fuzzing/workspace/output/mobus/seqgan",
        # r"/home/shenyuwang/paper_fuzzing/workspace/output/mqtt/seqgan"

    ]
    # data_resp_list = [
    #     r"/home/shenyuwang/paper_fuzzing/workspace/data/test1"  ## 存放一个个数据集的文件夹
    #     # r"/home/shenyuwang/paper_fuzzing/workspace/data/mqtt/cluster"  ## 存放一个个数据集的文件夹
    # ]
    # output_dir_root_list = [
    #     r"/home/shenyuwang/paper_fuzzing/workspace/output/test"
    #     # r"/home/shenyuwang/paper_fuzzing/workspace/output/mqtt/lstm"

    # ]
    for data_resp,output_dir_root in zip(data_resp_list,output_dir_root_list):
        data_set_list = retrun_allfiles_from_folder(data_resp)
        args_set = init_args_set(data_set_list, output_dir_root)
        for args in args_set:
            config_train = training_config()
            config_train.output_dir = args[0]
            if not os.path.exists(config_train.output_dir):  # os模块判断并创建
                os.mkdir(config_train.output_dir)

            config_train.positive_file = args[1]
            config_train.sequence_length = args[2]=args[2] if args[2]>20 else 20
            config_gen = generator_config()
            config_gen.sequence_length = args[2]=args[2] if args[2]>20 else 20
            config_dis = discriminator_config()
            config_dis.sequence_length = args[2]=args[2] if args[2]>20 else 20
            np.random.seed(config_train.seed)
            assert config_train.start_token == 17

            gen_data_loader = Gen_Data_loader(config_gen.gen_batch_size)
            # likelihood_data_loader = Gen_Data_loader(config_gen.gen_batch_size)
            dis_data_loader = Dis_dataloader(config_dis.dis_batch_size)

            # 构造graph
            tf.reset_default_graph()  # 告诉模型可以多次训练或者理解为清除之前的一些参数
            train_graph = tf.Graph()

            with tf.Session(graph=train_graph) as sess:
                
                generator = Generator(config=config_gen)
                generator.build()

                rollout_gen = rollout(config=config_gen)

                #Build target LSTM
                # target_params = pickle.load(open('save/target_params.pkl','rb'),encoding='iso-8859-1')
                # target_lstm = TARGET_LSTM(config=config_gen, params=target_params) # The oracle model


                # Build discriminator
                discriminator = Discriminator(config=config_dis)
                discriminator.build_discriminator()


                # Build optimizer op for pretraining
                pretrained_optimizer = tf.train.AdamOptimizer(config_train.gen_learning_rate)
                var_pretrained = [v for v in tf.trainable_variables() if 'teller' in v.name]
                gradients, variables = zip(
                    *pretrained_optimizer.compute_gradients(generator.pretrained_loss, var_list=var_pretrained))
                gradients, _ = tf.clip_by_global_norm(gradients, config_train.grad_clip)
                gen_pre_update = pretrained_optimizer.apply_gradients(zip(gradients, variables)) # 预训练更新




                sess.run(tf.global_variables_initializer())

                # generate_samples(sess,target_lstm,config_train.batch_size,config_train.generated_num,config_train.positive_file)
                gen_data_loader.create_batches(config_train.positive_file,config_train.sequence_length)

                # log = open('save/experiment-log.txt','w')
                print('Start pre-training generator....')

                # log.write('pre-training...\n')
                train_start_time = time.time()
                for epoch in range(config_train.pretrained_epoch_num): # 预训练生成器
                    gen_data_loader.reset_pointer()
                    for it in range(gen_data_loader.num_batch):
                        batch = gen_data_loader.next_batch()
                        input_mask_tmp = np.where(np.logical_not(np.equal(16, batch)), 1 * np.ones_like(batch), batch)
                        input_mask = np.where(np.equal(16, input_mask_tmp), 0 * np.ones_like(input_mask_tmp),
                                              input_mask_tmp)
                        # print(batch)
                        # print(input_mask)
                        _,g_loss = sess.run([gen_pre_update,generator.pretrained_loss],feed_dict={generator.input_seqs_pre:batch, # 预训练更新
                                                                                                  generator.input_seqs_mask:input_mask})




                print('Start pre-training discriminator...')
                for t in range(config_train.dis_update_time_pre):
                    print("Times: " + str(t))
                    generate_samples(sess,generator,config_train.batch_size,config_train.generated_num,config_train.negative_file)
                    dis_data_loader.load_train_data(config_train.positive_file,config_train.negative_file,config_train.sequence_length)
                    for _ in range(3):
                        dis_data_loader.reset_pointer()
                        for it in range(dis_data_loader.num_batch): # 整个epoch
                            x_batch,y_batch = dis_data_loader.next_batch()
                            feed_dict = {
                                discriminator.input_x : x_batch,
                                discriminator.input_y : y_batch,
                                discriminator.dropout_keep_prob : config_dis.dis_dropout_keep_prob
                            }
                            _ = sess.run(discriminator.train_op,feed_dict)



                # Build optimizer op for adversarial training
                train_adv_opt = tf.train.AdamOptimizer(config_train.gen_learning_rate)
                gradients, variables = zip(*train_adv_opt.compute_gradients(generator.gen_loss_adv, var_list=var_pretrained))
                gradients, _ = tf.clip_by_global_norm(gradients, config_train.grad_clip)
                train_adv_update = train_adv_opt.apply_gradients(zip(gradients, variables))

                # Initialize global variables of optimizer for adversarial training
                uninitialized_var = [e for e in tf.global_variables() if e not in tf.trainable_variables()]
                init_vars_uninit_op = tf.variables_initializer(uninitialized_var)
                sess.run(init_vars_uninit_op)

                # Start adversarial training
                step = 0
                fig_d_loss_trains = np.zeros([config_train.total_batch])
                fig_g_loss_trains = np.zeros([config_train.total_batch])
                fig_time = np.zeros([config_train.total_batch])
                ###############################################更新生成器#########################################################
                for total_batch in range(config_train.total_batch):
                    gen_loss = 0
                    dis_loss = 0
                    for iter_gen in range(config_train.gen_update_time): # 生成器的更新次数
                        samples = sess.run(generator.sample_word_list_reshpae)

                        feed = {'pred_seq_rollout:0':samples}
                        reward_rollout = []
                        for iter_roll in range(config_train.rollout_num):
                            rollout_list = sess.run(rollout_gen.sample_rollout_step,feed_dict=feed)
                            # np.vstack 它是垂直（按照行顺序）的把数组给堆叠起来。
                            rollout_list_stack = np.vstack(rollout_list)
                            reward_rollout_seq = sess.run(discriminator.ypred_for_auc,feed_dict={
                                discriminator.input_x:rollout_list_stack,discriminator.dropout_keep_prob:1.0
                            })
                            reward_last_tok = sess.run(discriminator.ypred_for_auc,feed_dict={
                                discriminator.input_x:samples,discriminator.dropout_keep_prob:1.0
                            })
                            reward_allseq = np.concatenate((reward_rollout_seq,reward_last_tok),axis=0)[:,1]
                            reward_tmp = []
                            for r in range(config_gen.gen_batch_size):
                                reward_tmp.append(reward_allseq[range(r,config_gen.gen_batch_size * config_gen.sequence_length,config_gen.gen_batch_size)])

                            reward_rollout.append(np.array(reward_tmp))

                        rewards = np.sum(reward_rollout,axis = 0) / (1.0 * config_train.rollout_num)
                        _,gen_loss = sess.run([train_adv_update,generator.gen_loss_adv],feed_dict={generator.input_seqs_adv:samples,
                                                                                                       generator.rewards:rewards})
                        step = step + 1
                    # if total_batch % config_train.test_per_epoch == 0 or total_batch == config_train.total_batch - 1: # 或者是总epoch
                    #     generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.eval_file)
                    #     likelihood_data_loader.create_batches(config_train.eval_file)
                    #     test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                    #     buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                    #     print ('total_batch: ', total_batch, 'test_loss: ', test_loss)
                    #     log.write(buffer)
            #######################################################更新鉴别器########################################################

                    for _ in range(config_train.dis_update_time_adv): # 鉴别器的更新次数
                        generate_samples(sess,generator,config_train.batch_size,config_train.generated_num,config_train.negative_file)
                        dis_data_loader.load_train_data(config_train.positive_file,config_train.negative_file,config_train.sequence_length)

                        for _ in range(3):
                            dis_data_loader.reset_pointer()
                            for it in range(dis_data_loader.num_batch):
                                x_batch,y_batch = dis_data_loader.next_batch()
                                feed = {
                                    discriminator.input_x:x_batch,
                                    discriminator.input_y:y_batch,
                                    discriminator.dropout_keep_prob:config_dis.dis_dropout_keep_prob
                                }
                                _,dis_loss = sess.run([discriminator.train_op,discriminator.loss],feed)
            #####################################################记录epoch信息######################################
                    fig_d_loss_trains[total_batch] = dis_loss
                    fig_g_loss_trains[total_batch] = gen_loss
                    fig_time[total_batch] = time.time() - train_start_time

                    if (total_batch+1) % config_train.test_per_epoch == 0:
                        save_model(sess, dis_data_loader.i2w, config_train.batch_size, generator, config_train.save_generated_num, total_batch,
                                   config_train.output_dir, "epoch"+str(total_batch+1), step, config_train.model_name,
                                   train_start_time, total_batch+1, fig_d_loss_trains, fig_g_loss_trains, fig_time)

    # sess.close()
    # log.close()

def generate_data():
    data_resp_list = [
        r"/home/shenyuwang/paper_fuzzing/workspace/data/coap/testing_set" , ## 存放一个个数据集的文件夹
        # r"/home/shenyuwang/paper_fuzzing/workspace/data/modbus/testing_set",
        # r"/home/shenyuwang/paper_fuzzing/workspace/data/mqtt/testing_set"  ## 存放一个个数据集的文件夹
    ]
    output_dir_root_list = [
        r"/home/shenyuwang/paper_fuzzing/workspace/output/coap/seqgan",
        # r"/home/shenyuwang/paper_fuzzing/workspace/output/mobus/seqgan",
        # r"/home/shenyuwang/paper_fuzzing/workspace/output/mqtt/seqgan"

    ]
    for data_resp,output_dir_root in zip(data_resp_list,output_dir_root_list):
        data_set_list = retrun_allfiles_from_folder(data_resp)
        args_set = init_args_set_testing(data_set_list, output_dir_root)
        for args in args_set:            
            for epoch_model_file in os.listdir(args[0]):
                epoch_model_file_path =  args[0] + os.path.sep + epoch_model_file
                config_train = training_config()
                config_train.output_dir = args[0]
                if not os.path.exists(config_train.output_dir):  # os模块判断并创建
                    # os.mkdir(config_train.output_dir)
                    continue

                config_train.positive_file = args[1]
                config_train.sequence_length = args[2]=args[2] if args[2]>20 else 20
                config_gen = generator_config()
                config_gen.sequence_length = args[2]=args[2] if args[2]>20 else 20
                config_dis = discriminator_config()
                config_dis.sequence_length = args[2]=args[2] if args[2]>20 else 20
                np.random.seed(config_train.seed)
                assert config_train.start_token == 17

                
                gen_data_loader = Gen_Data_loader(config_gen.gen_batch_size)
                # likelihood_data_loader = Gen_Data_loader(config_gen.gen_batch_size)
                dis_data_loader = Dis_dataloader(config_dis.dis_batch_size)

                # 构造graph
                tf.reset_default_graph()  # 告诉模型可以多次训练或者理解为清除之前的一些参数
                train_graph = tf.Graph()

                with tf.Session(graph=train_graph) as sess:
                    train_start_time = time.time()
                    train_end_time = train_start_time + 5 * 60
                    generator = Generator(config=config_gen)
                    generator.build()

                    rollout_gen = rollout(config=config_gen)

                    #Build target LSTM
                    # target_params = pickle.load(open('save/target_params.pkl','rb'),encoding='iso-8859-1')
                    # target_lstm = TARGET_LSTM(config=config_gen, params=target_params) # The oracle model


                    # Build discriminator
                    discriminator = Discriminator(config=config_dis)
                    discriminator.build_discriminator()





                    load(sess,epoch_model_file_path,config_train.model_name)

                    # sess.run(tf.global_variables_initializer())

                    # generate_samples(sess,target_lstm,config_train.batch_size,config_train.generated_num,config_train.positive_file)
                    # gen_data_loader.create_batches(config_train.positive_file,config_train.sequence_length)

                    generate_real_sample(sess, dis_data_loader.i2w, epoch_model_file, config_train.output_dir, generator, config_train.save_generated_num, config_train.batch_size,train_end_time)

                    # save_model(sess, dis_data_loader.i2w, config_train.batch_size, generator, config_train.save_generated_num, total_batch,
                    #                 config_train.output_dir, "epoch"+str(total_batch+1), step, config_train.model_name,
                    #                 train_start_time, total_batch+1, fig_d_loss_trains, fig_g_loss_trains, fig_time)
def main(unused_argv):
    # training()
    generate_data()



if __name__ == '__main__':
    try:
        tf.app.run()
    except Exception as e:
        with open("error.txt", "a") as error_f:
            error_f.write(
                "#######################################################本次运行报错信息如下：#######################################################\n")
            traceback.print_exc(file=error_f)

            for ele1 in tf.trainable_variables():
                error_f.write(ele1.name + '\n')
            error_f.write("\n")
    



