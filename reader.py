
from queue import Queue
from threading import Thread
import random

def padding_seq(seq):
    max_len = 0
    for s in seq:
        if max_len < len(s):
            max_len = len(s)
    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        seq[i] += [0 for i in range(l)]
    return seq


def encode_text(words, vocab_indices):
    return [vocab_indices[word] for word in words if word in vocab_indices]


def decode_text(labels, vocabs, end_token = '</s>'):
    results = []
    for idx in labels:
        word = vocabs[idx]
        if word == end_token:
            return ' '.join(results)
        results.append(word)
    return ' '.join(results)


def read_vocab(vocab_file):
     f = open(vocab_file, 'rb')
     vocabs = [line.decode('utf8')[:-1] for line in f]
     f.close()
     return vocabs


class SeqReader():
    def __init__(self, input_file, target_file, vocab_file, batch_size,
            queue_size = 2048, worker_size = 2, end_token = '</s>',
            padding = True, max_len = 50):
        self.input_file = input_file
        self.target_file = target_file
        self.end_token = end_token
        self.batch_size = batch_size
        self.padding = padding
        self.max_len = max_len
        # self.vocabs = read_vocab(vocab_file) + [end_token]
        self.vocabs = read_vocab(vocab_file)
        self.vocab_indices = dict((c, i) for i, c in enumerate(self.vocabs))
        self.data_queue = Queue(queue_size)
        self.worker_size = worker_size
        with open(self.input_file) as f:
            for i, l in enumerate(f):
                pass
            f.close()
            self.single_lines = i+1
        self.data_size = int(self.single_lines / batch_size)
        self.data_pos = 0
        self._init_reader()


    def start(self):
        return
    '''
        for i in range(self.worker_size):
            t = Thread(target=self._init_reader())
            t.daemon = True
            t.start()
    '''


    def read(self):
        while True:
            yield self.data[self.data_pos]
            self.data_pos += 1
            if self.data_pos >= len(self.data):
                random.shuffle(self.data)
                self.data_pos = 0


    def _init_reader(self):
        def init_batch():
            return {'in_seq': [],
                    'in_seq_len': [],
                    'target_seq': [],
                    'target_seq_len': []}
        self.data = []
        i = 0
        batch = init_batch()
        input_f = open(self.input_file, 'rb')
        target_f = open(self.target_file, 'rb')
        for input_line in input_f:
            if i == self.batch_size:
                if self.padding:
                    batch['in_seq'] = padding_seq(batch['in_seq'])
                    batch['target_seq'] = padding_seq(batch['target_seq'])
                self.data.append(batch)
                i = 0
                batch = init_batch()
            i += 1
            input_line = input_line.decode('utf-8')[:-1]
            target_line = target_f.readline().decode('utf-8')[:-1]
            input_words = input_line.split(' ')
            if len(input_words) >= self.max_len:
                input_words = input_words[:self.max_len-1]
            input_words.append(self.end_token)
            target_words = target_line.split(' ')
            if len(target_words) >= self.max_len:
                target_words = target_words[:self.max_len-1]
            target_words = ['<s>',] + target_words
            target_words.append(self.end_token)
            in_seq = encode_text(input_words, self.vocab_indices)
            target_seq = encode_text(target_words, self.vocab_indices)
            batch['in_seq'].append(in_seq)
            batch['in_seq_len'].append(len(in_seq))
            batch['target_seq'].append(target_seq)
            batch['target_seq_len'].append(len(target_seq)-1)
        input_f.close()
        target_f.close()

        '''

        def worker_func():
            i = 0
            batch = init_batch()
            while True:
                input_f = open(self.input_file, 'rb')
                target_f = open(self.target_file, 'rb')
                for input_line in input_f:
                    if i == self.batch_size:
                        if self.padding:
                            batch['in_seq'] = padding_seq(batch['in_seq'])
                            batch['target_seq'] = padding_seq(batch['target_seq'])
                        self.data_queue.put(batch)
                        i = 0
                        batch = init_batch()
                    i += 1
                    input_line = input_line.decode('utf-8')[:-1]
                    target_line = target_f.readline().decode('utf-8')[:-1]
                    input_words = input_line.split(' ')
                    if len(input_words) >= self.max_len:
                        input_words = input_words[:self.max_len-1]
                    input_words.append(self.end_token)
                    target_words = target_line.split(' ')
                    if len(target_words) >= self.max_len:
                        target_words = target_words[:self.max_len-1]
                    target_words = ['<s>',] + target_words
                    target_words.append(self.end_token)
                    in_seq = encode_text(input_words, self.vocab_indices)
                    target_seq = encode_text(target_words, self.vocab_indices)
                    batch['in_seq'].append(in_seq)
                    batch['in_seq_len'].append(len(in_seq))
                    batch['target_seq'].append(target_seq)
                    batch['target_seq_len'].append(len(target_seq)-1)
                input_f.close()
                target_f.close()
        return worker_func
        '''
