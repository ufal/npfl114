#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
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

        # TODO(tagger_crf): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.

        # TODO(tagger_crf): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocab_size()` call returning the number of unique words in the mapping.

        # TODO(tagger_crf): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the embedded words, **summing** the outputs of forward and backward RNNs.

        # TODO(tagger_crf): Add a final classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that **no activation** should
        # be used, the CRF operations will take care of it. Also do not forget to use
        # `tf.keras.layers.TimeDistributed`.

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)

        # We compile the model without loss, because `train_step` will directly call
        # the `selt.crf_loss` method.
        self.compile(optimizer=tf.optimizers.Adam(),
                     metrics=[self.SpanLabelingF1Metric(train.tags.word_mapping.get_vocabulary(), name="f1")])

        # TODO(tagger_crf): Create `self._crf_weights`, a trainable zero-initialized tf.float32 matrix variable
        # of size [number of unique train tags, number of unique train.tags], using `self.add_weight`.
        self._crf_weights = self.add_weight(...)

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

    def crf_loss(self, gold_labels, logits):
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CRF loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CRF loss must be RaggedTensors"

        # TODO: Implement the CRF loss computation manually, without using the `tfa.text` methods.
        # You can count on the fact that all training sentences contain at least 2 words.
        #
        # The following remarks might come handy:
        # - Custom RNN cells can be implemented by deriving from tf.keras.layers.AbstractRNNCell
        #   and definining at least `state_size`, `output_size` and `call`:
        #
        #     class CRFCell(tf.keras.layers.AbstractRNNCell):
        #         @property
        #         def output_size(self):
        #             # Return output dimensionality as a scalar number
        #         @property
        #         def state_size(self):
        #             # Return state dimensionality as either a scalar number or a vector
        #         def call(self, inputs, states):
        #             # Given inputs from the current timestep and states from the previous one,
        #             # return an (outputs, new_states) pair. Note that `states` and `new_states`
        #             # must always be a tuple of tensors, even if there is only a single state.
        #
        #   Such a cell can then be used by the `tf.keras.layers.RNN` layer. If you want to
        #   specify a different initial state then all zeros, pass it as `initial_state` argument
        #   along with the inputs.
        #
        # - RaggedTensors cannot be directly indexed in the ragged dimension, but they can be sliced.
        #   For example, to skip the first word in gold_labels, you can call
        #     gold_labels[:, 1:]
        #   but to get the first word in gold_labels, you cannot use
        #     gold_labels[:, 0]
        #   If you really require indexing in the ragged dimension, convert them to dense tensors.
        #
        # - To index a (possible ragged) tensor with another (possible ragged) tensor,
        #   `tf.gather` and `tf.gather_nd` can be used. If is useful fo pay attention
        #   to the `batch_dims` argument of these calls.
        raise NotImplementedError()

    def crf_decode(self, logits):
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # TODO(tagger_crf): Perform CRF decoding using `tfa.text.crf_decode`. Convert the
        # logits analogously as in `crf_loss`. Finally, convert the result
        # to a ragged tensor.
        #
        # Note: ignore the warning generated by tensorflow_addons/text/crf.py:540.
        # It does not apply to us, because we are passing a regular tensor to it.
        predictions = ...

        assert isinstance(predictions, tf.RaggedTensor)
        return predictions

    # We override the `train_step` method, because:
    # - computing losses on RaggedTensors is not supported in TF 2.4
    # - we do not want to evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.crf_loss(y, y_pred)
            if self.losses: # Add regularization losses if present
                loss += tf.math.add_n(self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    # We override `predict_step` to run CRF decoding during prediction
    def predict_step(self, data):
        if isinstance(data, tuple): data = data[0]
        y_pred = self(data, training=False)
        y_pred = self.crf_decode(y_pred)
        return y_pred

    # We override `test_step` to use `predict_step` to obtain CRF predictions.
    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(data)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    class SpanLabelingF1Metric(tf.metrics.Metric):
        """Keras-like metric evaluating span labeling F1-score of RaggedTensors."""
        def __init__(self, tags, name="span_labeling_f1", dtype=None):
            super().__init__(name, dtype)
            self._tags = tags
            self._counts = self.add_weight("counts", shape=[3], initializer=tf.initializers.Zeros(), dtype=tf.int64)

        def reset_states(self):
            self._counts.assign([0] * 3)

        def classify_spans(self, y_true, y_pred, sentence_limits):
            sentence_limits = set(sentence_limits)
            spans_true, spans_pred = set(), set()
            for spans, labels in [(spans_true, y_true), (spans_pred, y_pred)]:
                span = None
                for i, label in enumerate(self._tags[label] for label in labels):
                    if span and (label.startswith(("O", "B")) or i in sentence_limits):
                        spans.add((start, i, span))
                        span = None
                    if label.startswith("B"):
                        span, start = label[2:], i
                if span:
                    spans.add((start, len(labels), span))
            return np.array([len(spans_true & spans_pred), len(spans_pred - spans_true), len(spans_true - spans_pred)], np.int64)

        def update_state(self, y_true, y_pred, sample_weight=None):
            assert isinstance(y_true, tf.RaggedTensor) and isinstance(y_pred, tf.RaggedTensor)
            assert sample_weight is None, "sample_weight currently not supported"
            counts = tf.numpy_function(self.classify_spans, (y_true.values, y_pred.values, y_true.row_limits()), tf.int64)
            self._counts.assign_add(counts)

        def result(self):
            tp, fp, fn = self._counts[0], self._counts[1], self._counts[2]
            return tf.math.divide_no_nan(tf.cast(2 * tp, tf.float32), tf.cast(2 * tp + fp + fn, tf.float32))

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
    morpho = MorphoDataset("czech_cnec", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args, morpho.train)

    # TODO(tagger_crf): Construct dataset for training, which should contain pairs of
    # - tensor of string words (forms) as input
    # - tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(forms, lemmas, tags):
        raise NotImplementedError()

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(tagging_dataset)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    logs = network.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[network.tb_callback])

    # Return train loss and dev set accuracy for ReCodEx to validate
    return logs.history["loss"][-1], logs.history["val_f1"][-1]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
