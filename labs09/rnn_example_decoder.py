from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

import contrib_seq2seq

# Simple decoder for training
def decoder_fn_train(encoder_state, output_fn, input_fn, name=None):
    def decoder_fn(time, cell_state, next_id, cell_output, context_state):
        cell_output = output_fn(cell_output)
        if cell_state is None:  # first call, return encoder_state
            cell_state = encoder_state
        cell_input = input_fn(tf.squeeze(next_id, [1]), cell_state)
        return (None, cell_state, cell_input, cell_output, context_state)

    return decoder_fn

# Simple decoder for inference
def decoder_fn_inference(encoder_state, output_fn, input_fn,
                         beginning_of_word, end_of_word, maximum_length):
    batch_size = tf.shape(encoder_state)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        cell_output = output_fn(cell_output)
        if cell_state is None:
            cell_state = encoder_state
            next_id = tf.tile([beginning_of_word], [batch_size])
            done = tf.zeros([batch_size], dtype=tf.bool)
        else:
            next_id = tf.argmax(cell_output, 1)
            done = tf.equal(next_id, end_of_word)
            done = tf.cond(tf.greater_equal(time, maximum_length), # return true if time >= maxlen
                           lambda: tf.ones([batch_size], dtype=tf.bool),
                           lambda: done)
        next_input = input_fn(next_id, cell_state)
        return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn

LETTERS, BOW, EOW = 6, 0, 1
LETTERS_DIM = 20

embeddings = tf.get_variable("embeddings", shape=[LETTERS, LETTERS_DIM], dtype=tf.float32)
input = tf.placeholder(tf.int32, [None, None])

# Output functio (makes logits out of rnn outputs)
def output_fn(cell_output):
    if cell_output is None:
        return tf.zeros([LETTERS], tf.float32) # only used for shape inference
    else:
        return tf_layers.linear(cell_output, num_outputs=LETTERS, scope="rnn_output")

# Input function (makes rnn input from word id and cell state)
def input_fn(next_id, cell_state):
    return tf.nn.embedding_lookup(embeddings, next_id)

# Training
GRU_DIM = 20
MAX_GEN_LEN = 99
EPOCHS = 1000

cell = tf.nn.rnn_cell.GRUCell(GRU_DIM)

with tf.variable_scope("rnn_decoding"):
    input_lens = tf.tile(tf.gather(tf.shape(input), [1]) - 1, tf.gather(tf.shape(input), [0]))
    training_logits, states = \
        contrib_seq2seq.dynamic_rnn_decoder(cell,
                                            decoder_fn_train(cell.zero_state(tf.shape(input)[0], tf.float32), output_fn, input_fn),
                                            inputs=tf.expand_dims(input, -1),
                                            sequence_length=input_lens)

with tf.variable_scope("rnn_decoding", reuse=True):
    inference_logits, states = \
        contrib_seq2seq.dynamic_rnn_decoder(cell,
                                            decoder_fn_inference(cell.zero_state(1, tf.float32), output_fn, input_fn, BOW, EOW, MAX_GEN_LEN))
    output = tf.argmax(inference_logits, 2)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(training_logits, input[:, 1:]))
training = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as s:
    s.run(tf.initialize_all_variables())
    # Training
    for i in range(EPOCHS):
        s.run(training, {input: [[BOW, 2, 3, 2, 3, 4, 5, 4, 5, EOW]]})
    # Simple inference
    print(s.run(output))
