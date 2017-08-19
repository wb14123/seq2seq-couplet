
from model import Model

m = Model(
        '/data/dl-data/iwslt-2015/train.vi',
        '/data/dl-data/iwslt-2015/train.en',
        '/data/dl-data/iwslt-2015/tst2013.vi',
        '/data/dl-data/iwslt-2015/tst2013.en',
        '/data/dl-data/iwslt-2015/vocabs',
        num_units=512, layers=2, dropout=0.2,
        batch_size=64, learning_rate=0.001, output_dir='./output_iwslt')

m.train(5000000)
