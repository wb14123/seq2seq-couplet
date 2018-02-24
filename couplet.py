
from model import Model

m = Model(
        '/data/dl-data/couplet/train/in.txt',
        '/data/dl-data/couplet/train/out.txt',
        '/data/dl-data/couplet/test/in.txt',
        '/data/dl-data/couplet/test/out.txt',
        '/data/dl-data/couplet/vocabs',
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001,
        output_dir='/data/dl-data/models/tf-lib/output_couplet',
        restore_model=False)

m.train(5000000)
