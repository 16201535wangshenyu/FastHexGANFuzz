import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import os


def load_data(filename, w2i, max_length):
    data = []
    with open(filename, 'r') as f:
        lines = [list(line[:-1]) for line in f]
        random.shuffle(lines)
        for line in lines:
            # if len(line) < max_length:
            #     line += 'u' * (max_length - len(line))
            data.append(np.array([w2i[word.lower()] for word in line], dtype=int))
    return np.array(data)


def pad_sentence_batch(sentence_batch, pad_int, eof_int, is_target):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    if is_target:
        batch_data = []
        for sentence in sentence_batch:
            sentence = sentence.tolist()
            sentence.append(eof_int)
            sentence = np.array(sentence)
            batch_data.append(sentence)
        sentence_batch = np.array(batch_data)

    max_sentence = max([len(sentence) for sentence in sentence_batch])
    result = [sentence.tolist() + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    return result


def inf_train_gen(data, batch_size, w2i):
    np.random.shuffle(data)
    n_batches = len(data) // batch_size
    data = data.tolist()
    data.append(data[0])# 多增加一条data，以防止溢出
    data = np.array(data)

    ##计算target_seq_len

    for i in range(n_batches):
        source_data = data[i * batch_size:(i + 1) * batch_size]
        target_data = data[i * batch_size + 1:(i + 1) * batch_size + 1]
        source_data = np.array(pad_sentence_batch(source_data, w2i['u'], w2i['n'], False))
        target_data = np.array(pad_sentence_batch(target_data, w2i['u'], w2i['n'], True))
        ##计算source_seq_len
        input_mask_tmp = np.where(np.logical_not(np.equal(16, source_data)), 1 * np.ones_like(source_data), source_data)
        input_mask = np.where(np.equal(16, input_mask_tmp), 0 * np.ones_like(input_mask_tmp),
                              input_mask_tmp)
        source_data_lens = np.sum(input_mask, axis=1)
        ##计算target_seq_len
        input_mask_tmp = np.where(np.logical_not(np.equal(16, target_data)), 1 * np.ones_like(target_data), target_data)
        input_mask = np.where(np.equal(16, input_mask_tmp), 0 * np.ones_like(input_mask_tmp),
                              input_mask_tmp)
        target_seq_lens = np.sum(input_mask, axis=1)
        # target_seq_lens = target_seq_lens - np.ones_like(target_seq_lens)
        yield source_data, target_data, source_data_lens, target_seq_lens


def save_gen_samples(generate_data, i2w, prefix, outputdir):
    res = []
    for data in generate_data:
        for i in range(len(data)):
            l = [i2w[int(index)] for index in data[i]]
            while 'n' in l:
                l.remove('n')
            while 'u' in l:
                l.remove('u')
            while 's' in l:
                l.remove('s')
            res.append(l)

    with open(outputdir + os.path.sep + prefix + '_generate_data.txt', 'w+') as f:
        for i in range(len(res)):
            f.write(''.join(res[i]) + '\n')
    # print("Saved epoch " + str(e) + " generate data.")


def translate(data, i2w):
    res = []
    for i in range(len(data)):
        l = [i2w[int(index)] for index in data[i]]
        # while 'n' in l:
        #     l.remove('n')
        # while 'u' in l:
        #    l.remove('u')
        # while 's' in l:
        #     l.remove('s')
        res.append(l)
    for i in range(len(res)):
        print(''.join(res[i]))
    return res


def read_vocab(file_name):
    word_to_id = {}
    id_to_word = {}
    with open(file_name, 'r') as f:
        for line in f:
            word, index = line.strip().split('\t')
            iden = int(index)
            word_to_id[word] = iden
            id_to_word[iden] = word
    return word_to_id, id_to_word


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def seq2index(seq, w2i):
    data = []
    for i in seq:
        data.append([w2i[word] for word in seq[i]])
    return data





