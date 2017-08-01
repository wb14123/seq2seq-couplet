

import tensorflow as tf
from tensorflow.contrib import rnn


def getLayeredCell(layer_size, num_units, input_keep_prob,
        output_keep_prob=1.0):
    return rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.LSTMCell(num_units),
        input_keep_prob, output_keep_prob) for i in range(layer_size)])


def bi_encoder(embed_input, in_seq_len, num_units, layer_size, input_keep_prob):
    # encode input into a vector
    bi_layer_size = int(layer_size / 2)
    encode_cell_fw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    encode_cell_bw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = encode_cell_fw,
            cell_bw = encode_cell_bw,
            inputs = embed_input,
            sequence_length = in_seq_len,
            dtype = embed_input.dtype,
            time_major = False)

    # concat encode output and state
    encoder_output = tf.concat(bi_encoder_output, -1)
    encoder_state = []
    for layer_id in range(bi_layer_size):
        encoder_state.append(bi_encoder_state[0][layer_id])
        encoder_state.append(bi_encoder_state[1][layer_id])
    encoder_state = tuple(encoder_state)
    return encoder_output, encoder_state


def attention_decoder_cell(encoder_output, in_seq_len, num_units, layer_size,
        input_keep_prob):
    attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(num_units,
            encoder_output, in_seq_len, normalize = True)
    cell = getLayeredCell(layer_size, num_units, input_keep_prob)
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanim,
            attention_layer_size=num_units)
    return cell


def decoder_projection(output, output_size):
    return tf.layers.dense(output, output_size, activation=None,
            name='output_mlp')


def train_decoder(encoder_output, in_seq_len, target_seq, target_seq_len,
        encoder_state, num_units, layers, output_size, input_keep_prob):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
            layers, input_keep_prob)

    def _step(time, output, state, all_outputs):
        output, state = decoder_cell(tf.gather(target_seq, time), state)
        all_outputs = all_outputs.write(time, output)
        time = time + 1
        return time, output, state, all_outputs

    target_shape = tf.shape(target_seq)
    batch_size = target_shape[0]
    max_timestep = target_shape[1]
    target_seq = tf.transpose(target_seq, perm=[1, 0, 2])
    target_seq = tf.pad(target_seq, [[1, 0], [0, 0], [0, 0]], "CONSTANT")
    init_all_outputs = tf.TensorArray(dtype=target_seq.dtype,
            size=max_timestep, tensor_array_name='decoder_all_outputs')
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)
    init_output =tf.zeros([batch_size, num_units])
    time, output, state, all_output = tf.while_loop(
            cond = lambda time, *_: time < max_timestep,
            body = _step,
            loop_vars = (0, init_output, init_state, init_all_outputs))
    output = all_output.stack()
    output = decoder_projection(output, output_size)
    output = tf.transpose(output, perm=[1, 0, 2])
    return output


def infer_decoder(encoder_output, in_seq_len, encoder_state, num_units, layers,
        embedding, output_size, input_keep_prob):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
            layers, input_keep_prob)
    def _step(time, output, state, all_outputs):
        output, state = decoder_cell(output, state)
        output = tf.nn.softmax(decoder_projection(output, output_size))
        output = tf.to_int32(tf.argmax(output, 1))
        next_input = tf.nn.embedding_lookup(embedding, output,
                name="infer_embedding")
        all_outputs = all_outputs.write(time, output)
        time = time + 1
        return time, next_input, state, all_outputs

    input_shape = tf.shape(encoder_output)
    batch_size = input_shape[0]
    max_timestep = input_shape[1] * 2
    init_all_outputs = tf.TensorArray(dtype=tf.int32,
            size=max_timestep, tensor_array_name='decoder_all_outputs')
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)
    init_output =tf.zeros([batch_size, num_units])
    time, output, state, all_output = tf.while_loop(
            cond = lambda time, *_: time < max_timestep,
            body = _step,
            loop_vars = (0, init_output, init_state, init_all_outputs))
    output = all_output.stack()
    output = tf.transpose(output, perm=[1, 0])
    return output


def seq2seq(in_seq, in_seq_len, target_seq, target_seq_len, vocab_size,
        num_units, layers, dropout):
    in_shape = tf.shape(in_seq)
    batch_size = in_shape[0]
    input_keep_prob = 1 - dropout

    # embedding input and target sequence
    with tf.device('/cpu:0'):
        embedding = tf.get_variable(
                name = 'embedding',
                shape = [vocab_size, num_units])
    embed_input = tf.nn.embedding_lookup(embedding, in_seq, name='embed_input')

    # encode and decode
    encoder_output, encoder_state = bi_encoder(embed_input, in_seq_len,
            num_units, layers, input_keep_prob)
    if target_seq != None:
        embed_target = tf.nn.embedding_lookup(embedding, target_seq,
                name='embed_target')
        decoder_output = train_decoder(encoder_output, in_seq_len, embed_target,
                target_seq_len, encoder_state, num_units, layers, vocab_size,
                input_keep_prob)
        return decoder_output
    else:
        return infer_decoder(encoder_output, in_seq_len, encoder_state,
                num_units, layers, embedding, vocab_size, input_keep_prob)


def seq_loss(output, target, seq_len):
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
            labels=target)
    loss_mask = tf.sequence_mask(seq_len, tf.shape(output)[1])
    cost = cost * tf.to_float(loss_mask)
    return tf.reduce_mean(cost)

