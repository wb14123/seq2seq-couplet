
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from model import Model

app = Flask(__name__)
CORS(app)

m = Model(
        '/data/dl-data/couplet/train/in.txt',
        '/data/dl-data/couplet/train/out.txt',
        '/data/dl-data/couplet/test/in.txt',
        '/data/dl-data/couplet/test/out.txt',
        '/data/dl-data/couplet/vocabs',
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir='/data/dl-data/models/tf-lib/output_couplet',
        restore_model=True, init_train=False, init_infer=True)

@app.route('/chat/couplet/<in_str>')
def chat_couplet(in_str):
    if len(in_str) == 0 or len(in_str) > 20:
        output = u'您的输入太长了'
    else:
        output = m.infer(' '.join(in_str))
        output = ''.join(output.split(' '))
    return jsonify({'output': output})
