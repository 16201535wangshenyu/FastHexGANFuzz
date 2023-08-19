import numpy as np
import random


class Gen_Data_loader():
    """
    主要用于Generator的数据生成器。在Generator的预训练 以及 计算Generator和Oracle model的相似性的时候使用
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []
        self.w2i, self.i2w = self.read_vocab(r'save/vocab')

    def read_vocab(self, file_name):
        word_to_id = {}
        id_to_word = {}
        with open(file_name, 'r') as f:
            for line in f:
                # print(line)
                word, index = line.strip().split('\t')
                iden = int(index)
                word_to_id[word] = index
                id_to_word[iden] = word
        return word_to_id, id_to_word

    def create_batches(self, data_file, max_length):
        self.token_stream = []
        with open(data_file, 'r') as f:
            lines = [list(line[:-1]) for line in f]
            random.shuffle(lines)
            for line in lines:
                if len(line) < max_length:
                    line += 'u' * (max_length - len(line))
                self.token_stream.append(np.array([self.w2i[word.lower()] for word in line], dtype=int))

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        # 截取刚刚好的batch
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        # 使用np的split函数切分batch
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0





class Dis_dataloader():
    """
    主要用于Discriminator的训练。
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.w2i, self.i2w = self.read_vocab(r'save/vocab')

    def load_train_data(self, positive_file, negative_file, max_length):
        positive_examples = []
        negative_examples = []

        with open(positive_file, 'r') as f:
            lines = [list(line[:-1]) for line in f]
            random.shuffle(lines)
            lines = lines[:(10000//64)*64]
            for line in lines:
                if len(line) < max_length:
                    line += 'u' * (max_length - len(line))
                positive_examples.append([self.w2i[word.lower()] for word in line])

        with open(negative_file) as fin:
            for line in fin:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples + negative_examples)
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]

        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # shuffle the data
        # 如果传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本；
        # 而shuffle只是对一个矩阵进行洗牌，无返回值。 如果传入一个整数，它会返回一个洗牌后的arange。
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.batch_size * self.num_batch]
        self.labels = self.labels[:self.batch_size * self.num_batch]

        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def read_vocab(self, file_name):
        word_to_id = {}
        id_to_word = {}
        with open(file_name, 'r') as f:
            for line in f:
                word, index = line.strip().split('\t')
                iden = int(index)
                word_to_id[word] = index
                id_to_word[iden] = word
        return word_to_id, id_to_word

    def reset_pointer(self):
        self.pointer = 0


if __name__ == "__main__":
    data_loader = Dis_dataloader(64)
    data_loader.load_train_data(r'output/modbus/modbus_raw_data_46_6w.txt',r'save/generator_sample.txt',46)
    ret = data_loader.next_batch()
    print(ret)
    ret = data_loader.next_batch()
    print(ret)