
import tensorflow as tf

import reader


def test_couplet_loss():
    input_file = '/data/dl-data/couplet/test/in.txt'
    output_file = '/data/dl-data/couplet/test/out.txt'
    vocab_file = '/data/dl-data/couplet/vocabs'
    batch_size = 4
    test_reader = reader.SeqReader(input_file, target_file, vocab_file,
            batch_size)
    test_reader.start()
    data = test_reader.read()
