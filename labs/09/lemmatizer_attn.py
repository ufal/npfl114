#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import decoder
from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, args, num_source_chars, num_target_chars):
        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()

                # TODO(lemmatizer_noattn): Define
                # - source_embeddings as a masked embedding layer of source chars into args.cle_dim dimensions

                # TODO: Define
                # - source_rnn as a bidirectional GRU with args.rnn_dim units, returning _whole sequences_, summing opposite directions

                # TODO(lemmatizer_noattn): Define
                # - target_embedding as an unmasked embedding layer of target chars into args.cle_dim dimensions
                # - target_rnn_cell as a GRUCell with args.rnn_dim units
                # - target_output_layer as a Dense layer into `num_target_chars`

                # TODO: Define
                # - attention_source_layer as a Dense layer with args.rnn_dim outputs
                # - attention_state_layer as a Dense layer with args.rnn_dim outputs
                # - attention_weight_layer as a Dense layer with 1 output

        self._model = Model()

        self._optimizer = tf.optimizers.Adam()
        # TODO(lemmatizer_noattn): Define self._loss as SparseCategoricalCrossentropy which processes _logits_ instead of probabilities
        self._metrics_training = {"loss": tf.metrics.Mean(), "accuracy": tf.metrics.SparseCategoricalAccuracy()}
        self._metrics_evaluation = {"accuracy": tf.metrics.Mean()}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def _append_eow(self, sequences):
        """Append EOW character after end every given sequence."""
        sequences_rev = tf.reverse_sequence(sequences, tf.reduce_sum(tf.cast(tf.not_equal(sequences, 0), tf.int32), axis=1), 1)
        sequences_rev_eow = tf.pad(sequences_rev, [[0, 0], [1, 0]], constant_values=MorphoDataset.Factor.EOW)
        return tf.reverse_sequence(sequences_rev_eow, tf.reduce_sum(tf.cast(tf.not_equal(sequences_rev_eow, 0), tf.int32), axis=1), 1)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 4, autograph=False)
    def train_batch(self, source_charseq_ids, source_charseqs, target_charseq_ids, target_charseqs):
        # TODO(lemmatizer_noattn): Modify target_charseqs by appending EOW; only the version with appended EOW is used from now on.

        with tf.GradientTape() as tape:
            # TODO(lemmatizer_noattn): Embed source charseqs
            # TODO: Run self._model.source_rnn on the embedded sequences, returning outputs in `source_encoded`.

            # Copy the source_encoded to corresponding batch places, and then flatten it
            source_mask = tf.not_equal(source_charseq_ids, 0)
            source_encoded = tf.boolean_mask(tf.gather(source_encoded, source_charseq_ids), source_mask)
            targets = tf.boolean_mask(tf.gather(target_charseqs, target_charseq_ids), source_mask)

            class DecoderTraining(decoder.BaseDecoder):
                @property
                def batch_size(self): raise NotImplemented() # TODO: Return batch size of self._source_encoded, using tf.shape
                @property
                def output_size(self): raise NotImplemented() # TODO(lemmatizer_noattn): Return number of the generated logits
                @property
                def output_dtype(self): return NotImplemented() # TODO(lemmatizer_noattn): Return the type of the generated logits

                def _with_attention(self, inputs, states):
                    # TODO: Compute the attention.
                    # - Take self._source_encoded and pass it through the self._model.attention_source_layer.
                    #   Because self._source_encoded does not change, you should in fact do it in `initialize`.
                    # - Pass `states` though self._model.attention_state_layer.
                    # - Sum the two outputs. However, the first has shape [a, b, c] and the second [a, c]. Therefore,
                    #   somehow expand the second to [a, b, c] first. (Hint: use broadcasting rules.)
                    # - Pass the sum through `tf.tanh`, then self._model.attention_weight_layer.
                    # - Then, run softmax on a suitable axis (the one corresponding to characters), generating `weights`.
                    # - Multiply `self._source_encoded` with `weights` and sum the result in the axis
                    #   corresponding to characters, generating `attention`. Therefore, `attention` is a a fixed-size
                    #   representation for every batch element, independently on how many characters had
                    #   the corresponding input forms.
                    # - Finally concatenate `inputs` and `attention` and return the result.

                def initialize(self, layer_inputs, initial_state=None):
                    self._model, self._source_encoded, self._targets = layer_inputs

                    # TODO(lemmatozer_noattn): Define `finished` as a vector of self.batch_size of `False` [see tf.fill].
                    # TODO(lemmatizer_noattn): Define `inputs` as a vector of self.batch_size MorphoDataset.Factor.BOW [see tf.fill],
                    # embedded using self._model.target_embedding
                    # TODO: Define `states` as the last words from self._source_encoded
                    # TODO: Pass `inputs` through `self._with_attention(inputs, states)`.
                    return finished, inputs, states

                def step(self, time, inputs, states):
                    # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self._model.target_rnn_cell, generating
                    # `outputs, [states]`.
                    # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self._model.target_output_layer,
                    # TODO(lemmatizer_noattn): Define `next_inputs` by embedding `time`-th words from `self._targets`.
                    # TODO(lemmatizer_noattn): Define `finished` as True if `time`-th word from `self._targets` is EOW, False otherwise.
                    # Again, no == or !=.
                    # TODO: Pass `inputs` through `self._with_attention(inputs, states)`.
                    return outputs, states, next_inputs, finished

            output_layer, _, _ = DecoderTraining()([self._model, source_encoded, targets])
            # TODO(lemmatizer_noattn): Compute loss. Use only nonzero `targets` as a mask.
        gradients = tape.gradient(loss, self._model.variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics_training.items():
                metric.reset_states()
                if name == "loss": metric(loss)
                else: metric(targets, output_layer, tf.not_equal(targets, 0))
                tf.summary.scalar("train/{}".format(name), metric.result())

        return tf.math.argmax(output_layer, axis=2)

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            # TODO(lemmatizer_noattn): Call train_batch, storing results in `predictions`.

            form, gold_lemma, system_lemma = "", "", ""
            for i in batch[dataset.FORMS].charseqs[1]:
                if i: form += dataset.data[dataset.FORMS].alphabet[i]
            for i in range(len(batch[dataset.LEMMAS].charseqs[1])):
                if batch[dataset.LEMMAS].charseqs[1][i]:
                    gold_lemma += dataset.data[dataset.LEMMAS].alphabet[batch[dataset.LEMMAS].charseqs[1][i]]
                    system_lemma += dataset.data[dataset.LEMMAS].alphabet[predictions[0][i]]
            print(float(self._metrics_training["accuracy"].result()), form, gold_lemma, system_lemma)


    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 2, autograph=False)
    def predict_batch(self, source_charseq_ids, source_charseqs):
        # TODO(lemmatizer_noattn)(train_batch): Embed source charseqs
        # TODO(train_batch): Run self._model.source_rnn on the embedded sequences, returning outputs in `source_encoded`.

        # Copy the source_encoded to corresponding batch places, and then flatten it
        source_mask = tf.not_equal(source_charseq_ids, 0)
        source_encoded = tf.boolean_mask(tf.gather(source_encoded, source_charseq_ids), source_mask)

        class DecoderPrediction(decoder.BaseDecoder):
            @property
            def batch_size(self): raise NotImplemented() # TODO(train_batch): Return batch size of self._source_encoded, using tf.shape
            @property
            def output_size(self): raise NotImplemented() # TODO(lemmatizer_noattn): Return 1 because we are returning directly the predictions
            @property
            def output_dtype(self): return NotImplemented() # TODO(lemmatizer_noattn): Return tf.int32 because the predictions are integral

            def _with_attention(self, inputs, states):
                # TODO: A copy of _with_attention from train_batch; you can of course
                # move the definition to a place where it can be reused in both places.

            def initialize(self, layer_inputs, initial_state=None):
                self._model, self._source_encoded = layer_inputs

                # TODO(lemmatizer_noattn)(train_batch): Define `finished` as a vector of self.batch_size of `False` [see tf.fill].
                # TODO(lemmatizer_noattn)(train_batch): Define `inputs` as a vector of self.batch_size MorphoDataset.Factor.BOW [see tf.fill],
                # embedded using self._model.target_embedding
                # TODO(train_batch): Define `states` as the last words from self._source_encoded
                # TODO(train_batch): Pass `inputs` through `self._with_attention(inputs, states)`.
                return finished, inputs, states

            def step(self, time, inputs, states):
                # TODO(lemmatizer_noattn)(train_batch): Pass `inputs` and `[states]` through self._model.target_rnn_cell, generating
                # `outputs, [states]`.
                # TODO(lemmatizer_noattn)(train_batch): Overwrite `outputs` by passing them through self._model.target_output_layer,
                # TODO(lemmatizer_noattn): Overwirte `outputs` by passing them through `tf.argmax` on suitable axis and with
                # `output_type=tf.int32` parameter.
                # TODO(lemmatizer_noattn): Define `next_inputs` by embedding the `outputs`
                # TODO(lemmatizer_noattn): Define `finished` as True if `outputs` are EOW, False otherwise. [No == or !=].
                # TODO: Pass `inputs` through `self._with_attention(inputs, states)`.
                return outputs, states, next_inputs, finished

        predictions, _, _ = DecoderPrediction(maximum_iterations=tf.shape(source_charseqs)[1] + 10)([self._model, source_encoded])
        return predictions

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 4, autograph=False)
    def evaluate_batch(self, source_charseq_ids, source_charseqs, target_charseq_ids, target_charseqs):
        # Predict
        predictions = self.predict_batch(source_charseq_ids, source_charseqs)

        # Append EOW to target_charseqs and copy them to corresponding places and flatten it
        target_charseqs = self._append_eow(target_charseqs)
        targets = tf.boolean_mask(tf.gather(target_charseqs, target_charseq_ids), tf.not_equal(source_charseq_ids, 0))

        # Compute accuracy, but on the whole sequences
        mask = tf.cast(tf.not_equal(targets, 0), tf.int32)
        resized_predictions = tf.concat([predictions, tf.zeros_like(targets)], axis=1)[:, :tf.shape(targets)[1]]
        equals = tf.reduce_all(tf.equal(resized_predictions * mask, targets * mask), axis=1)
        self._metrics_evaluation["accuracy"](equals)

    def evaluate(self, dataset, dataset_name, args):
        for metric in self._metrics_evaluation.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            predictions = self.evaluate_batch(batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs,
                                              batch[dataset.LEMMAS].charseq_ids, batch[dataset.LEMMAS].charseqs)

        metrics = {name: float(metric.result()) for name, metric in self._metrics_evaluation.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.initializers.glorot_uniform(seed=42)
        tf.keras.utils.get_custom_objects()["orthogonal"] = lambda: tf.initializers.orthogonal(seed=42)
        tf.keras.utils.get_custom_objects()["uniform"] = lambda: tf.initializers.RandomUniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args,
                      num_source_chars=len(morpho.train.data[morpho.train.FORMS].alphabet),
                      num_target_chars=len(morpho.train.data[morpho.train.LEMMAS].alphabet))
    for epoch in range(args.epochs):
        network.train_epoch(morpho.train, args)
        metrics = network.evaluate(morpho.dev, "dev", args)
        print("Evaluation on {}, epoch {}: {}".format("dev", epoch + 1, metrics))

    metrics = network.evaluate(morpho.test, "test", args)
    with open("lemmatizer.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)
