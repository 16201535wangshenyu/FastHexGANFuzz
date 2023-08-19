import tensorflow as tf
import argparse
import sys
sys.path.append(r"/home/shenyuwang/paper_fuzzing/workspace/model/improved_wgan/v1_withoutFastRoPEAttention")
import time
from improved_wgan import *
from utils import *
import traceback

# 每一次修改不同的报文需要修改三个地方
# 1、--outputdir
# 2、--data_file
# 3、--seq_size

def parse_args(outputdir, data_file, seq_size):
    desc = "Tensorflow implementation of Self-Attention GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_file', type=str,
                        default=data_file)
    # 'data/mqtt/mqtt_all_lens_13w.txt'  'data/mqtt_58_60_4w.txt' #  'data/generated_write_single_register_1.txt'
    parser.add_argument('--vocab_file', type=str, default=r'vocab')
    parser.add_argument('--outputdir', type=str, default=outputdir)
    # 'output/mqtt/test20221022_1_mqtt_all_lens_13w'
    ####### z_dim value influences a lot to the performance
    parser.add_argument('--z_dim', type=int, default=50)

    parser.add_argument('--critic_iters', type=int, default=1)
    parser.add_argument('--LAMBDA', type=int, default=10)
    parser.add_argument('--seq_size', type=int, default=seq_size)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--d_model', type=int, default=80)
    parser.add_argument('--num_blocks', type=int, default=4)

    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--vocab_size', type=int, default=17)
    parser.add_argument('--dropout_rate', type=float, default=0.3) # 
    parser.add_argument('--d_ff', type=int, default=64)

    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='learning rate for discriminator')

    parser.add_argument('--g_beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--g_beta2', type=float, default=0.9, help='beta2 for Adam optimizer')

    parser.add_argument('--d_beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--d_beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    return parser.parse_args()


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


def main():
    data_resp_list = [
        # r"/home/shenyuwang/paper_fuzzing/workspace/data/test3" , ## 存放一个个数据集的文件夹
        # r"/home/shenyuwang/paper_fuzzing/workspace/data/modbus/trian_data",
        # r"/home/shenyuwang/paper_fuzzing/workspace/data/mqtt/testing_set",  ## 存放一个个数据集的文件夹
        r"/home/shenyuwang/paper_fuzzing/workspace/data/test2"
    ]
    output_dir_root_list = [
        # r"/home/shenyuwang/paper_fuzzing/workspace/output/coap/Fast_RopEGAN",
        # r"/home/shenyuwang/paper_fuzzing/workspace/output/mobus/Fast_RopEGAN",
        # r"/home/shenyuwang/paper_fuzzing/workspace/output/mqtt/Fast_RoPEGAN",
        r"/home/shenyuwang/paper_fuzzing/workspace/output/test2/Fast_RoPE_without"

    ]
    for data_resp,output_dir_root in zip(data_resp_list,output_dir_root_list):
    # data_resp = r"G:\研究生\研究生\科研\小论文\FCS-21046_Proof_hi\代码2\WGAN\data\coap\first_dataset\data_set_raw\data_prepare\cluster\coap_208_210_212_214_6.0w.txt"  ## 存放一个个数据集的文件夹
    # # data_resp = r"G:\研究生\研究生\科研\小论文\FCS-21046_Proof_hi\代码2\WGAN\data\coap\first_dataset\data_set_raw\data_prepare\coap_208_210_212_214_6.0w.txt"
    # output_dir_root = r"G:\研究生\研究生\科研\小论文\FCS-21046_Proof_hi\代码2\WGAN\output\coap\improved_wgan\train1"
    
        data_set_list = retrun_allfiles_from_folder(data_resp)
        args_set = init_args_set(data_set_list, output_dir_root)

        for args in args_set:
            # print("ars0:",args[0])
            # print("ars1:",args[1])
            # print("ars2:",args[2])
            args = parse_args(args[0], args[1], args[2])

            if args is None:
                exit()

            w2i, i2w = read_vocab(args.vocab_file)
            tf.reset_default_graph() # 告诉模型可以多次训练或者理解为清除之前的一些参数

            # sess = tf.Session()
            with tf.Session() as sess:
                gan = SA_GAN_SEQ(sess, args, w2i, i2w)
                #gan.build_model()
                data = load_data(args.data_file, w2i, args.seq_size)

                # show network architecture
                #show_all_variables()
                # result_txt = open("/home/shenyuwang/paper_fuzzing/workspace/model/improved_wgan/v1/result.txt","a")
                # for ele1 in tf.trainable_variables():
                #     print(ele1.name + '\n',file=result_txt)
                train_start_time = time.time()
                
                gan.train(data)
                print("Train epoch:", args.epoch)
                print("DONE. Time:", timedelta(seconds=time.time()-train_start_time))

            # sess.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        with open("error.txt", "a") as error_f:
            error_f.write(
                "#######################################################本次运行报错信息如下：#######################################################\n")
            traceback.print_exc(file=error_f)

            for ele1 in tf.trainable_variables():
                error_f.write(ele1.name + '\n')
            error_f.write("\n")


# inputs: Tensor("discriminator/Conv1d.1/transpose:0", shape=(64, 18, 17), dtype=float32)
# filters: <tf.Variable 'discriminator/Conv1d.1/Conv1d.1.Filters:0' shape=(1, 17, 80) dtype=float32_ref>
# graph: <tensorflow.python.framework.ops.Graph object at 0x0000015956BA9D68>


# inputs: Tensor("discriminator/Conv1d.1/transpose:0", shape=(64, 14, 17), dtype=float32)
# filters: <tf.Variable 'discriminator/Conv1d.1/Conv1d.1.Filters:0' shape=(1, 17, 80) dtype=float32_ref>