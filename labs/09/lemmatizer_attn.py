#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        self.source_mapping = train.forms.char_mapping
        self.target_mapping = train.lemmas.char_mapping
        self.target_mapping_inverse = type(self.target_mapping)(
            vocabulary=self.target_mapping.get_vocabulary(), invert=True)

        # TODO(lemmatizer_noattn): Define
        # - `self.source_embedding` as an embedding layer of source chars into `args.cle_dim` dimensions

        # TODO: Define
        # - `self.source_rnn` as a bidirectional GRU with `args.rnn_dim` units, returning **whole sequences**,
        #   summing opposite directions

        # TODO(lemmatizer_noattn): Then define
        # - `self.target_embedding` as an embedding layer of target chars into `args.cle_dim` dimensions
        # - `self.target_rnn_cell` as a GRUCell with `args.rnn_dim` units
        # - `self.target_output_layer` as a Dense layer into as many outputs as there are unique target chars

        # TODO: Define
        # - `self.attention_source_layer` as a Dense layer with `args.rnn_dim` outputs
        # - `self.attention_state_layer` as a Dense layer with `args.rnn_dim` outputs
        # - `self.attention_weight_layer` as a Dense layer with 1 output

        # Compile the model
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.Accuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    class DecoderTraining(tfa.seq2seq.BaseDecoder):
        def __init__(self, lemmatizer, *args, **kwargs):
            self.lemmatizer = lemmatizer
            super().__init__.__wrapped__(self, *args, **kwargs)

        @property
        def batch_size(self):
            # TODO(lemmatizer_noattn): Return the batch size of `self.source_states` as a *scalar* number;
            # use `tf.shape` to get the full shape and then extract the batch size.
            raise NotImplementedError()
        @property
        def output_size(self):
            # TODO(lemmatizer_noattn): Describe the size of a single decoder output (batch size and the
            # sequence length are not included) by returning
            #   tf.TensorShape(number of logits of each output element [lemma character])
            raise NotImplementedError()
        @property
        def output_dtype(self):
            # TODO(lemmatizer_noattn): Return the type of the decoder output (so the type of the
            # produced logits).
            raise NotImplementedError()

        def with_attention(self, inputs, states):
            # TODO: Compute the attention.
            # - Compute projected source states by passing `self.source_states` through the
            #   `self.lemmatizer.attention_source_layer`. Because `self.source_states` do not change,
            #   you should in fact precompute the projected source states once in `initialize`.
            # - Compute projected decoder state by passing `states` though `self.lemmatizer.attention_state_layer`.
            # - Sum the two projections. However, the first has shape [a, b, c] and the second [a, c]. Therefore,
            #   expand the second to [a, b, c] or [a, 1, c] (the latter works because of broadcasting rules).
            # - Pass the sum through `tf.tanh` and through the `self.lemmatizer.attention_weight_layer`.
            # - Then, run softmax on a suitable axis, generating `weights`.
            # - Multiply the original (non-projected) `self.source_states` with `weights` and sum the result
            #   in the axis corresponding to characters, generating `attention`. Therefore, `attention` is
            #   a fixed-size representation for every batch element, independently on how many characters
            #   the corresponding input forms had.
            # - Finally concatenate `inputs` and `attention` (in this order) and return the result.

            return tf.concat([inputs, attention], axis=1)

        def initialize(self, layer_inputs, initial_state=None, mask=None):
            self.source_states, self.targets = layer_inputs

            # TODO(lemmatizer_noattn): Define `finished` as a vector of self.batch_size of `False` [see tf.fill].

            # TODO(lemmatizer_noattn): Define `inputs` as a vector of self.batch_size of MorphoDataset.Factor.BOW,
            # embedded using self.lemmatizer.target_embedding

            # TODO: Define `states` as the representation of the first character
            # in `source_states`. The idea is that it is most relevant for generating
            # the first letter and contains all following characters via the backward RNN.

            # TODO: Pass `inputs` through `self.with_attention(inputs, states)`.

            return finished, inputs, states

        def step(self, time, inputs, states, training):
            # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,

            # TODO(lemmatizer_noattn): Define `next_inputs` by embedding `time`-th chars from `self.targets`.

            # TODO(lemmatizer_noattn): Define `finished` as a vector of booleans; True if the corresponding
            # `time`-th char from `self.targets` is `MorphoDataset.Factor.EOW`, False otherwise.

            # TODO: Pass `next_inputs` through `self.with_attention(next_inputs, states)`.

            return outputs, states, next_inputs, finished

    class DecoderPrediction(DecoderTraining):
        @property
        def output_size(self):
            # TODO(lemmatizer_noattn): Describe the size of a single decoder output (batch size and the
            # sequence length are not included) by returning a suitable
            # `tf.TensorShape` representing a *scalar* element, because we are producing
            # lemma character indices during prediction.
            raise NotImplementedError()
        @property
        def output_dtype(self):
            # TODO(lemmatizer_noattn): Return the type of the decoder output (i.e., target lemma character indices).
            raise NotImplementedError()

        def initialize(self, layer_inputs, initial_state=None, mask=None):
            # Use `initialize` from the `DecoderTraining`, passing None as `targets`.
            return super().initialize([layer_inputs, None], initial_state)

        def step(self, time, inputs, states, training):
            # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through `tf.argmax` on suitable axis and with
            # `output_type=tf.int32` parameter.

            # TODO(lemmatizer_noattn): Define `next_inputs` by embedding the `outputs`

            # TODO(lemmatizer_noattn): Define `finished` as a vector of booleans; True if the corresponding
            # prediction in `outputs` is `MorphoDataset.Factor.EOW`, False otherwise.

            # TODO(DecoderTraining): Pass `next_inputs` through `self.with_attention(next_inputs, states)`.

            return outputs, states, next_inputs, finished

    # If `targets` is given, we are in the teacher forcing mode.
    # Otherwise, we run in autoregressive mode.
    def call(self, inputs, targets=None):
        # Forget about sentence boundaries and instead consider
        # all valid form-lemma pairs as independent batch examples.
        #
        # Then, split the given forms into character sequences and map then
        # to their indices.
        source_charseqs = inputs.values
        source_charseqs = tf.strings.unicode_split(source_charseqs, "UTF-8")
        source_charseqs = self.source_mapping(source_charseqs)
        if targets is not None:
            # The targets are already mapped sequences of characters, so only
            # drop the sentence boundaries, and convert to a dense tensor
            # (the EOW correctly indicate end of lemma).
            target_charseqs = targets.values
            target_charseqs = target_charseqs.to_tensor()

        # TODO(lemmatizer_noattn): Embed source_charseqs using `source_embedding`

        # TODO: Run source_rnn on the embedded sequences, returning outputs in `source_states`.
        # However, convert the embedded sequences from a RaggedTensor to a dense Tensor first,
        # i.e., call the `source_rnn` with
        #   (source_embedded.to_tensor(), mask=tf.sequence_mask(source_embedded.row_lengths()))
        sources_states = None

        # Run the appropriate decoder. Note that the outputs of the decoders
        # are exactly the outputs of `tfa.seq2seq.dynamic_decode`.
        if targets is not None:
            # TODO(lemmatizer_noattn): Create a self.DecoderTraining by passing `self` to its constructor.
            # Then run it on `[source_states, target_charseqs]` input,
            # storing the first result in `output` and the third result in `output_lens`.
            raise NotImplementedError()
        else:
            # TODO(lemmatizer_noattn): Create a self.DecoderPrediction by using:
            # - `self` as first argument to its constructor
            # - `maximum_iterations=tf.cast(source_charseqs.bounding_shape(1) + 10, tf.int32)`
            #   as another argument, which indicates that the longest prediction
            #   must be at most 10 characters longer than the longest input.
            #
            # Then run it on `source_states`, storing the first result in `output`
            # and the third result in `output_lens`. Finally, because we do not want
            # to return the `[EOW]` symbols, decrease `output_lens` by one.
            raise NotImplementedError()

        # Reshape the output to the original matrix of lemmas
        # and explicitly set mask for loss and metric computation.
        output = tf.RaggedTensor.from_tensor(output, output_lens)
        output = inputs.with_values(output)
        return output

    def train_step(self, data):
        x, y = data

        # Convert `y` by splitting characters, mapping characters to ids using
        # `self.target_mapping` and finally appending `MorphoDataset.Factor.EOW`
        # to every sequence.
        y_targets = self.target_mapping(tf.strings.unicode_split(y.values, "UTF-8"))
        y_targets = tf.concat(
            [y_targets, tf.fill([y_targets.bounding_shape(0), 1],
                                tf.constant(MorphoDataset.Factor.EOW, tf.int64))], axis=-1)
        y_targets = y.with_values(y_targets)

        with tf.GradientTape() as tape:
            y_pred = self(x, targets=y_targets, training=True)
            loss = self.compute_loss(x, y_targets.values, y_pred.values)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    def predict_step(self, data):
        if isinstance(data, tuple): data = data[0]
        y_pred = self(data, training=False)
        y_pred = self.target_mapping_inverse(y_pred)
        y_pred = tf.strings.reduce_join(y_pred, axis=-1)
        return y_pred

    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(data)
        self.compiled_metrics.update_state(
            tf.ones_like(y.values, dtype=tf.int32), tf.cast(y_pred.values == y.values, tf.int32))
        return {m.name: m.result() for m in self.metrics if m.name != "loss"}


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences, add_bow_eow=True)

    # Create the model and train
    model = Model(args, morpho.train)

    # Construct dataset for lemmatizer training
    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(lambda example: (example["forms"], example["lemmas"]))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    # Callback showing intermediate results during training
    class ShowIntermediateResults(tf.keras.callbacks.Callback):
        def __init__(self, data):
            self._iterator = iter(data.repeat())
        def on_train_batch_end(self, batch, logs=None):
            if model.optimizer.iterations % 10 == 0:
                forms, lemmas = next(self._iterator)
                print(model.optimizer.iterations.numpy(),
                      *[repr(strings[0, 0].numpy().decode("utf-8"))
                        for strings in [forms, lemmas, model.predict_on_batch(forms[:1, :1])]])

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, verbose=2,
                     callbacks=[ShowIntermediateResults(dev), model.tb_callback])

    # Return all metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items()}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
