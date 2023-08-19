import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import os

import numpy as np


class DataLoad(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.vector_array = []
        self.vector_array_normal = []
        self.num_batch = 0
        self.vector_matrix = []
        self.pointer = 0

    def create_batches(self, file_name,w2i,max_length):
        self.vector_array = []  #每一行的数据读到这个 list中，最后通过np.array把他转变为矩阵

        with open(file_name, 'r') as f:
            lines = [list(line[:-1]) for line in f]
            random.shuffle(lines)
            for line in lines:
                if len(line) < max_length:
                    line += 'u' * (max_length - len(line))
                int_line = [w2i[word.lower()] for word in line]

                # print("int_line........")
                # print(int_line)
                self.vector_array.append(int_line)

        self.num_batch = int(len(self.vector_array) / self.batch_size)
        self.vector_array = self.vector_array[:self.num_batch * self.batch_size]#这一行好像没什么用，有用，应该是凑够整数个batch，不够的就丢掉了
        self.vector_array_normal = np.multiply(self.vector_array,1.0/16.0)
        self.vector_matrix = np.split(np.array(self.vector_array_normal), self.num_batch, 0)
        # print("load_noraml_data .....vector_matrix_normal...............")
        # print(self.vector_matrix)
        self.pointer = 0

    def next_batch(self):
        ret = self.vector_matrix[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


def load_data(filename, w2i, max_length):
    data = []
    with open(filename, 'r') as f:
        lines = [list(line[:-1]) for line in f]
        random.shuffle(lines)
        for line in lines:
            if len(line) < max_length:
                line += 'u' * (max_length - len(line))
            data.append(np.array([w2i[word.lower()] for word in line], dtype=int))
    return np.array(data)


def inf_train_gen(data, batch_size):
    np.random.shuffle(data)
    n_batches = len(data)//batch_size
    for i in range(n_batches):
        yield data[i * batch_size:(i+1) * batch_size]



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
            f.write(''.join(res[i])+'\n')
    # print("Saved epoch " + str(e) + " generate data.")

def translate(data, i2w):
    res = []
    for i in range(len(data)):
        l = [i2w[int(index)] for index in data[i]]
        # while 'n' in l:
        #     l.remove('n')
        #while 'u' in l:
        #    l.remove('u')
        # while 's' in l:
        #     l.remove('s')
        res.append(l)
    for i in range(len(res)):
        print(''.join(res[i]))
    return res

def generate_batch(data):
    return None

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

def make_noise(shape):
    noise = np.random.normal(size=shape)
    return noise

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def seq2index(seq, w2i):
    data = []
    for i in seq:
        data.append([w2i[word] for word in seq[i]])
    return data