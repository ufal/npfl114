#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Network(tf.keras.Model):
    def __init__(self, args, train):
        # Implement a one-layer RNN network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO: Map strings in `words` to indices by using the `word_mapping` of `train.forms`.

        # TODO: Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocab_size()` call returning the number of unique words in the mapping.

        # TODO: Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element. Then apply it in a bidirectional way on
        # the embedded words, **summing** the outputs of forward and backward RNNs.

        # TODO: Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train tags`. However, because we are applying the
        # the Dense layer to a ragged tensor, we need to wrap the Dense layer in
        # a tf.keras.layers.TimeDistributed.

        super().__init__(inputs=words, outputs=predictions)
        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=tf.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

    # Note that in TF 2.4, computing losses and metrics on RaggedTensors is not yet
    # supported (it will be in TF 2.5). Therefore, we override the `train_step` method
    # to support it, passing the "flattened" predictions and gold data to the loss
    # and metrics.
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

    # Analogously to `train_step`, we also need to override `test_step`.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args, morpho.train)

    # TODO: Construct dataset for training, which should contain pairs of
    # - ragged tensor of string words (forms) as input
    # - ragged tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(forms, lemmas, tags):
        raise NotImplementedError()

    train = morpho.train.dataset.map(tagging_dataset).apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dev = morpho.dev.dataset.map(tagging_dataset).apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    test = morpho.test.dataset.map(tagging_dataset).apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))

    network.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[network.tb_callback])

    test_logs = network.evaluate(test, return_dict=True)
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    # Return test set accuracy for ReCodEx to validate
    return test_logs["accuracy"]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
