import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
from datetime import timedelta


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []

    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = " ".join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def generate_real_sample(sess, i2w, prefix, outputdir, trainable_model, generated_num, batch_size,train_end_time):
    generated_samples = []
    total_frame_num = 0
    while True:
        generated_samples.extend(trainable_model.generate(sess))
        total_frame_num = total_frame_num + batch_size
        if (time.time()-train_end_time) >= 0:
                break
    with open(outputdir+os.path.sep+"gen_result.txt","w") as wf:
        wf.write("total_frame_num:"+str(total_frame_num)+"\n")
        wf.write("exceed_seconds:"+str(time.time()-train_end_time)+"\n")



    # res = []
    # for data in generated_samples:
        
    #     l = [i2w[int(index)] for index in data]
    #     while 'u' in l:
    #         l.remove('u')
    #     while 's' in l:
    #         l.remove('s')
    #     res.append(l)
    # with open(outputdir + os.path.sep + prefix + '_generate_data.txt', 'w+') as f:
    #     for i in range(len(res)):
    #         f.write(''.join(res[i]) + '\n')


# def target_loss(sess, generated_samples, data_loader):
#     """
#     使用target lstm评估
#     :param sess:
#     :param generated_samples:
#     :param data_loader:
#     :return:
#     """
#     nll = []
#     data_loader.reset_pointer()
#
#     for it in range(data_loader.num_batch):
#         batch = data_loader.next_batch()
#         g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
#         nll.append(g_loss)
#
#     return np.mean(nll)


def draw_result_picture(total_batch, fig_d_loss_trains, fig_g_loss_trains, folder, fig_time, output_dir):
    save_dir = output_dir + os.path.sep + folder
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    lns1 = ax1.plot(np.arange(total_batch), fig_g_loss_trains, 'g', label="Generator_loss")
    lns2 = ax2.plot(np.arange(total_batch), fig_d_loss_trains, 'r', label="Critic_loss")

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Generator_loss')
    ax2.set_ylabel('Critic_loss')
    # 合并图例

    lns = lns1 + lns2
    labels = ["Generator_loss", "Critic_loss", ]

    plt.legend(lns, labels, loc=7)
    plt.savefig(save_dir + "/figure.png")
    plt.savefig(save_dir + "/figure.pdf", bbox_inches='tight', pad_inches=0.01)
    # plt.show()

    ##############################Critic_loss ###############################
    plt.figure()
    plt.plot(fig_time, fig_d_loss_trains)
    plt.xlabel('Wallclock time (in seconds)')
    plt.ylabel('Critic_loss')
    plt.savefig(save_dir + '/figure_time_Critic_loss.png')
    # plt.show()

    ##############################Generator_loss ###############################
    plt.figure()
    plt.plot(fig_time, fig_g_loss_trains)
    plt.xlabel('Wallclock time (in seconds)')
    plt.ylabel('Generator_loss')
    plt.savefig(save_dir + '/figure_time_Generator_loss.png')
    # plt.show()

    # save data
    np.save(save_dir + '/fig_d_loss_trains.npy', np.array(fig_d_loss_trains))
    np.save(save_dir + '/fig_g_loss_trains.npy', np.array(fig_g_loss_trains))
    np.save(save_dir + '/fig_time.npy', np.array(fig_time))
    plt.close()


def save(sess, checkpoint_dir, step, model_name):  # step代表参数更新了多少次
    checkpoint_dir = os.path.join(checkpoint_dir, model_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tf.train.Saver().save(sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)


def load(sess, checkpoint_dir, model_name):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_name)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        tf.train.Saver().restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


############替换#############################
def save_final_info(outputdir, folder, file_name, content):
    write_folder = outputdir + os.path.sep + folder
    if not os.path.exists(write_folder):
        os.mkdir(write_folder)
    result_file_path = write_folder + os.path.sep + file_name
    with open(result_file_path, "w+") as result_file:
        result_file.write(content)


def save_model(sess, i2w, batch_size, trainable_model, generated_num, e, outputdir, folder, step, model_name,
               train_start_time, batch, fig_d_loss_trains, fig_g_loss_trains, fig_time):
    # 保存样本
    generate_real_sample(sess, i2w, folder, outputdir, trainable_model, generated_num, batch_size)

    ## 保存时间####
    write_to_content = "current_epoch:" + str(e + 1) + "\n"
    write_to_content += "After training, total time:" + str(
        timedelta(seconds=time.time() - train_start_time)) + "\n"

    save_final_info(outputdir, folder, "result.txt", write_to_content)
    ## 保存model e+1 当前的epoch ###
    save(sess, outputdir + os.path.sep + folder, step, model_name)
    ##画图,保存图片信息######

    fig_d_loss_trains_t = fig_d_loss_trains[:batch]
    fig_g_loss_trains_t = fig_g_loss_trains[:batch]
    fig_time_t = fig_time[:batch]

    draw_result_picture(batch, fig_d_loss_trains_t, fig_g_loss_trains_t, folder, fig_time_t, outputdir)
