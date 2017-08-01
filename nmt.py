
from model import Model

m = Model(
        '/data/dl-data/wmt-2016/train.tok.clean.bpe.32000.en',
        '/data/dl-data/wmt-2016/train.tok.clean.bpe.32000.de',
        '/data/dl-data/wmt-2016/newstest2016.tok.en',
        '/data/dl-data/wmt-2016/newstest2016.tok.de',
        '/data/dl-data/wmt-2016/vocabs.de-en',
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001, output_dir='./output',
        param_histogram=True, restore_model=True)

m.train(5000000)
