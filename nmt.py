
from model import Model

m = Model(
        '/data/dl-data/wmt-2016/train.tok.clean.bpe.32000.de',
        '/data/dl-data/wmt-2016/train.tok.clean.bpe.32000.en',
        '/data/dl-data/wmt-2016/newstest2016.tok.bpe.32000.de',
        '/data/dl-data/wmt-2016/newstest2016.tok.bpe.32000.en',
        '/data/dl-data/wmt-2016/vocab.bpe.32000.bk',
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir='/data/dl-data/models/tf-lib/output_nmt',
        restore_model=True)

m.train(5000000, start=102500)
