from gevent import monkey
monkey.patch_all()

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from model import Model
from gevent.wsgi import WSGIServer
from logging.handlers import RotatingFileHandler
import logging

app = Flask(__name__)
CORS(app)

vocab_file = '/data/dl-data/couplet/vocabs'
model_dir = '/data/dl-data/models/tf-lib/output_couplet_prod'


def log_setup():
    log_handler = RotatingFileHandler(
        "/logs/service.log",
        maxBytes=1024*1024*20,  # 20M per log file
        backupCount=1000 # keep 1000 log files
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(process)d] - [%(threadName)s]: %(message)s')
    log_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)
    logging.info("Inited logger")

log_setup()

m = Model(
        None, None, None, None, vocab_file,
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)

SPLIT_CHARS = ['，', '、', ',', '.', '。', '!', '！', '?', '？', ' ']

def manual_correct_result(in_str, outputs, scores):
    for i in range(len(outputs)):
        output = outputs[i]
        scores[i] -= abs(len(in_str) - len(output))
        length = min(len(in_str), len(output))
        for j in range(length):
            for k in range(j, length):
                if (in_str[j] == in_str[k]) != (output[j] == output[k]):
                    scores[i] -= 1
        for j in range(length):
            for k in range(length):
                if output[k] not in SPLIT_CHARS and (in_str[j] == output[k]):
                    scores[i] -= 1
    return outputs, scores


def sort_outputs(outputs, scores):
    new_scores, new_outputs = zip(*sorted(zip(scores, outputs), reverse=True))
    return list(new_outputs), list(new_scores)


def infer(in_str):
    if len(in_str) == 0 or len(in_str) > 50:
        return [u'您的输入太长了'], []
    else:
        model_outputs, model_scores = m.infer(' '.join(in_str))
        model_scores = model_scores.tolist()
        unsorted_outputs, unsorted_scores = manual_correct_result(
                in_str, model_outputs, model_scores)
        output, score = sort_outputs(unsorted_outputs, unsorted_scores)
        logging.info('上联：%s；下联：%s ; score: %s' % (
            in_str, output, score))
        return output, score


@app.route('/chat/couplet/<in_str>')
def chat_couplet(in_str):
    output, score = infer(in_str)
    return jsonify({'output': output[0]})


@app.route('/v0.2/couplet/<in_str>')
def chat_couplet_v2(in_str):
    output, score = infer(in_str)
    return jsonify({'output': output, 'score': score})


http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
