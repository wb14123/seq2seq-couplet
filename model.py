
import tensorflow as tf
import seq2seq
import bleu
import reader
from os import path
import random


class Model():

    def __init__(self, train_input_file, train_target_file,
            test_input_file, test_target_file, vocab_file,
            num_units, layers, dropout,
            batch_size, learning_rate, output_dir,
            save_step = 100, eval_step = 1000,
            param_histogram=False, restore_model=False,
            init_train=True, init_infer=False):
        self.num_units = num_units
        self.layers = layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_step = save_step
        self.eval_step = eval_step
        self.param_histogram = param_histogram
        self.restore_model = restore_model
        self.init_train = init_train
        self.init_infer = init_infer

        if init_train:
            self.train_reader = reader.SeqReader(train_input_file,
                    train_target_file, vocab_file, batch_size)
            self.train_reader.start()
            self.train_data = self.train_reader.read()
            self.eval_reader = reader.SeqReader(test_input_file, test_target_file,
                    vocab_file, batch_size)
            self.eval_reader.start()
            self.eval_data = self.eval_reader.read()

        self.model_file = path.join(output_dir, 'model.ckpl')
        self.log_writter = tf.summary.FileWriter(output_dir)

        if init_train:
            self._init_train()
            self._init_eval()

        if init_infer:
            self.infer_vocabs = reader.read_vocab(vocab_file)
            self.infer_vocab_indices = dict((c, i) for i, c in
                    enumerate(self.infer_vocabs))
            self._init_infer()
            self.reload_infer_model()


    def gpu_session_config(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return config


    def _init_train(self):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self.train_in_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.train_in_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_target_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.train_target_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])
            output = seq2seq.seq2seq(self.train_in_seq, self.train_in_seq_len,
                    self.train_target_seq, self.train_target_seq_len,
                    len(self.train_reader.vocabs),
                    self.num_units, self.layers, self.dropout)
            self.train_output = tf.argmax(tf.nn.softmax(output), 2)
            self.loss = seq2seq.seq_loss(output, self.train_target_seq,
                    self.train_target_seq_len)
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                        gradients, 0.5)
            self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                ).apply_gradients(zip(clipped_gradients,params))
            if self.param_histogram:
                for v in tf.trainable_variables():
                    tf.summary.histogram('train_' + v.name, v)
            tf.summary.scalar('loss', self.loss)
            self.train_summary = tf.summary.merge_all()
            self.train_init = tf.global_variables_initializer()
            self.train_saver = tf.train.Saver()
        self.train_session = tf.Session(graph=self.train_graph,
                config=self.gpu_session_config())


    def _init_eval(self):
        self.eval_graph = tf.Graph()
        with self.eval_graph.as_default():
            self.eval_in_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.eval_in_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.eval_output = seq2seq.seq2seq(self.eval_in_seq,
                    self.eval_in_seq_len, None, None,
                    len(self.eval_reader.vocabs),
                    self.num_units, self.layers, self.dropout)
            if self.param_histogram:
                for v in tf.trainable_variables():
                    tf.summary.histogram('eval_' + v.name, v)
            self.eval_summary = tf.summary.merge_all()
            self.eval_saver = tf.train.Saver()
        self.eval_session = tf.Session(graph=self.eval_graph,
                config=self.gpu_session_config())


    def _init_infer(self):
        self.infer_graph = tf.Graph()
        with self.infer_graph.as_default():
            self.infer_in_seq = tf.placeholder(tf.int32, shape=[1, None])
            self.infer_in_seq_len = tf.placeholder(tf.int32, shape=[1])
            self.infer_output = seq2seq.seq2seq(self.infer_in_seq,
                    self.infer_in_seq_len, None, None,
                    len(self.infer_vocabs),
                    self.num_units, self.layers, self.dropout)
            self.infer_saver = tf.train.Saver()
        self.infer_session = tf.Session(graph=self.infer_graph,
                config=self.gpu_session_config())



    def train(self, epochs, start=0):
        if not self.init_train:
            raise Exception('Train graph is not inited!')
        with self.train_graph.as_default():
            if path.isfile(self.model_file + '.meta') and self.restore_model:
                print("Reloading model file before training.")
                self.train_saver.restore(self.train_session, self.model_file)
            else:
                self.train_session.run(self.train_init)
            total_loss = 0
            for step in range(start, epochs):
                data = next(self.train_data)
                in_seq = data['in_seq']
                in_seq_len = data['in_seq_len']
                target_seq = data['target_seq']
                target_seq_len = data['target_seq_len']
                output, loss, train, summary = self.train_session.run(
                        [self.train_output, self.loss, self.train_op, self.train_summary],
                        feed_dict={
                            self.train_in_seq: in_seq,
                            self.train_in_seq_len: in_seq_len,
                            self.train_target_seq: target_seq,
                            self.train_target_seq_len: target_seq_len})
                total_loss += loss
                self.log_writter.add_summary(summary, step)
                if step % self.save_step == 0:
                    self.train_saver.save(self.train_session, self.model_file)
                    print("Saving model. Step: %d, loss: %f" % (step,
                        total_loss / self.save_step))
                    # print sample output
                    sid = random.randint(0, self.batch_size-1)
                    input_text = reader.decode_text(in_seq[sid],
                        self.eval_reader.vocabs)
                    output_text = reader.decode_text(output[sid],
                            self.train_reader.vocabs)
                    target_text = reader.decode_text(target_seq[sid],
                            self.train_reader.vocabs).split(' ')[1:]
                    target_text = ' '.join(target_text)
                    print('******************************')
                    print('src: ' + input_text)
                    print('output: ' + output_text)
                    print('target: ' + target_text)
                if step % self.eval_step == 0:
                    bleu_score = self.eval(step)
                    print("Evaluate model. Step: %d, score: %f, loss: %f" % (
                        step, bleu_score, total_loss / self.save_step))
                    eval_summary = tf.Summary(value=[tf.Summary.Value(
                        tag='bleu', simple_value=bleu_score)])
                    self.log_writter.add_summary(eval_summary, step)
                if step % self.save_step == 0:
                    total_loss = 0


    def eval(self, train_step):
        with self.eval_graph.as_default():
            self.eval_saver.restore(self.eval_session, self.model_file)
            bleu_score = 0
            target_results = []
            output_results = []
            for step in range(0, self.eval_reader.data_size):
                data = next(self.eval_data)
                in_seq = data['in_seq']
                in_seq_len = data['in_seq_len']
                target_seq = data['target_seq']
                target_seq_len = data['target_seq_len']
                outputs = self.eval_session.run(
                        self.eval_output,
                        feed_dict={
                            self.eval_in_seq: in_seq,
                            self.eval_in_seq_len: in_seq_len})
                for i in range(len(outputs)):
                    output = outputs[i]
                    target = target_seq[i]
                    output_text = reader.decode_text(output,
                            self.eval_reader.vocabs).split(' ')
                    target_text = reader.decode_text(target[1:],
                            self.eval_reader.vocabs).split(' ')
                    prob = int(self.eval_reader.data_size * self.batch_size / 10)
                    target_results.append([target_text])
                    output_results.append(output_text)
                    if random.randint(1, prob) == 1:
                        print('====================')
                        input_text = reader.decode_text(in_seq[i],
                                self.eval_reader.vocabs)
                        print('src:' + input_text)
                        print('output: ' + ' '.join(output_text))
                        print('target: ' + ' '.join(target_text))
            return bleu.compute_bleu(target_results, output_results)[0] * 100


    def reload_infer_model(self):
        with self.infer_graph.as_default():
            self.infer_saver.restore(self.infer_session, self.model_file)


    def infer(self, text):
        if not self.init_infer:
            raise Exception('Infer graph is not inited!')
        with self.infer_graph.as_default():
            in_seq = reader.encode_text(text.split(' ') + ['</s>',],
                    self.infer_vocab_indices)
            in_seq_len = len(in_seq)
            outputs = self.infer_session.run(self.infer_output,
                    feed_dict={
                        self.infer_in_seq: [in_seq],
                        self.infer_in_seq_len: [in_seq_len]})
            output = outputs[0]
            output_text = reader.decode_text(output, self.infer_vocabs)
            return output_text
