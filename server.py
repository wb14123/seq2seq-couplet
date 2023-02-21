
print("Program starting ...", flush=True)

import sys

from gevent import monkey

print("Patching monkey ...", flush=True)
monkey.patch_all()

print("Importing library ...", flush=True)
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from gevent.pywsgi import WSGIServer
from logging.handlers import RotatingFileHandler
import logging

print("Importing model ...", flush=True)
from model import Model


print("Creating Flask app ...", flush=True)
app = Flask(__name__)
CORS(app)

vocab_file = '/data/dl-data/couplet/vocabs'
model_dir = '/data/dl-data/models/tf-lib/output_couplet_prod'


print("Setting up logging ...", flush=True)

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


SPLIT_CHARS = ['，', '、', ',', '.', '。', '!', '！', '?', '？', ' ']
CENSOR_WORDS_DICT = "/data/censor_words.txt"

logging.info("Loading censor words...")
with open(CENSOR_WORDS_DICT, encoding='utf-8') as censor_words_file:
    censor_words = [word[:-1] for word in censor_words_file.readlines()]
logging.info("Loaded %s censor_words" % (len(censor_words)))


m = Model(
        None, None, None, None, vocab_file,
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)


def all_same(s):
    if len(s) <= 1:
        return True
    for i in range(1, len(s)):
        if s[i] not in SPLIT_CHARS and s[i] != s[0]:
            return False
    return True


def manual_correct_result(in_str, outputs, scores):
    is_all_same = all_same(in_str)
    for i in range(len(outputs)):
        if is_all_same:
            scores[i] -= 100
            continue
        output = outputs[i]
        scores[i] -= abs(len(in_str) - len(output))
        length = min(len(in_str), len(output))
        for censor_word in censor_words:
            if in_str.find(censor_word) >= 0 or output.find(censor_word) >= 0:
                scores[i] -= 1000
                break

        for j in range(length):
            for k in range(j, length):
                if (in_str[j] == in_str[k]) != (output[j] == output[k]):
                    scores[i] -= 10
        for j in range(length):
            for k in range(length):
                if output[k] not in SPLIT_CHARS and (in_str[j] == output[k]):
                    scores[i] -= 10
        if length > 0:
            scores[i] = scores[i] - ((length ** -3) * 100)
        else:
            scores[i] = -100
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


logging.info("Starting server ...")
http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
