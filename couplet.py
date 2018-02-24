
from model import Model

m = Model(
        '/data/dl-data/couplet/train/in.txt',
        '/data/dl-data/couplet/train/out.txt',
        '/data/dl-data/couplet/test/in.txt',
        '/data/dl-data/couplet/test/out.txt',
        '/data/dl-data/couplet/vocabs',
        num_units=1024,
        layers=4,
        dropout=0.2,
        batch_size=32,
        learning_rate=0.00001,
        # repeat_weight=20000,
        # cross_repeat_weight=2000,
        repeat_weight=0,
        cross_repeat_weight=0,
        output_dir='./output_couplet_repeat',
        restore_model=True,
        )

m.train(5000000, start=570500)
