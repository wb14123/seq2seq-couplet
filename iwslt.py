
from model import Model

m = Model(
        '/data/dl-data/iwslt15-google/train.en',
        '/data/dl-data/iwslt15-google/train.vi',
        '/data/dl-data/iwslt15-google/tst2013.en',
        '/data/dl-data/iwslt15-google/tst2013.vi',
        '/data/dl-data/iwslt15-google/vocab.en.bk',
        num_units=512, layers=2, dropout=0.2,
        batch_size=128, learning_rate=0.001, output_dir='./output_iwslt')

m.train(5000000)
