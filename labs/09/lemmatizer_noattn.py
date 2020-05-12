#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from morpho_dataset import MorphoDataset

class Network:
    class Lemmatizer(tf.keras.Model):
        def __init__(self, args, num_source_chars, num_target_chars):
            super().__init__()

            # TODO: Define
            # - `source_embedding` as a masked embedding layer of source chars into args.cle_dim dimensions
            # - `source_rnn` as a bidirectional GRU with args.rnn_dim units, returning only the last output
            #   (i.e., return_sequences=False), summing opposite directions

            # - `target_embedding` as an unmasked embedding layer of target chars into args.cle_dim dimensions
            # - `target_rnn_cell` as a GRUCell with args.rnn_dim units
            # - `target_output_layer` as a Dense layer into `num_target_chars`

        class DecoderTraining(tfa.seq2seq.BaseDecoder):
            def __init__(self, lemmatizer, *args, **kwargs):
                self.lemmatizer = lemmatizer
                super().__init__.__wrapped__(self, *args, **kwargs)

            @property
            def batch_size(self):
                 # TODO: Return the batch size of self.source_states, using tf.shape
                raise NotImplementedError()
            @property
            def output_size(self):
                 # TODO: Return `tf.TensorShape(number of logits per each output element)`
                 # By output element we mean characters.
                raise NotImplementedError()
            @property
            def output_dtype(self):
                 # TODO: Return the type of the logits
                raise NotImplementedError()

            def initialize(self, layer_inputs, initial_state=None):
                self.source_states, self.targets = layer_inputs

                # TODO: Define `finished` as a vector of self.batch_size of `False` [see tf.fill].
                # TODO: Define `inputs` as a vector of self.batch_size of MorphoDataset.Factor.BOW,
                #   embedded using self.lemmatizer.target_embedding
                # TODO: Define `states` as self.source_states
                return finished, inputs, states

            def step(self, time, inputs, states, training):
                # TODO: Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
                #   which returns `(outputs, [states])`.
                # TODO: Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
                # TODO: Define `next_inputs` by embedding `time`-th chars from `self.targets`.
                # TODO: Define `finished` as True if `time`-th char from `self.targets` is EOW, False otherwise.
                return outputs, states, next_inputs, finished

        class DecoderPrediction(DecoderTraining):
            @property
            def output_size(self):
                 # TODO: Return `tf.TensorShape()` describing a scalar element,
                 # because we are generating scalar predictions now.
                raise NotImplementedError()
            @property
            def output_dtype(self):
                 # TODO: Return the type of the generated predictions
                raise NotImplementedError()

            def initialize(self, layer_inputs, initial_state=None):
                self.source_states = layer_inputs

                # TODO(DecoderTraining): Use the same initialization as in DecoderTraining.
                return finished, inputs, states

            def step(self, time, inputs, states, training):
                # TODO(DecoderTraining): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
                #   which returns `(outputs, [states])`.
                # TODO(DecoderTraining): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
                # TODO: Overwrite `outputs` by passing them through `tf.argmax` on suitable axis and with
                #   `output_type=tf.int32` parameter.
                # TODO: Define `next_inputs` by embedding the `outputs`
                # TODO: Define `finished` as True if `outputs` are EOW, False otherwise.
                return outputs, states, next_inputs, finished

        def call(self, inputs):
            # If `inputs` is a list of two elements, we are in the teacher forcing mode.
            # Otherwise, we run in autoregressive mode.
            if isinstance(inputs, list) and len(inputs) == 2:
                source_charseqs, target_charseqs = inputs
            else:
                source_charseqs, target_charseqs = inputs, None
            source_charseqs_shape = tf.shape(source_charseqs)

            # Get indices of valid lemmas and reshape the `source_charseqs`
            # so that it is a list of valid sequences, instead of a
            # matrix of sequences, some of them padding ones.
            valid_words = tf.cast(tf.where(source_charseqs[:, :, 0] != 0), tf.int32)
            source_charseqs = tf.gather_nd(source_charseqs, valid_words)
            if target_charseqs is not None:
                target_charseqs = tf.gather_nd(target_charseqs, valid_words)

            # TODO: Embed source_charseqs using `source_embedding`
            # TODO: Run source_rnn on the embedded sequences, returning outputs in `source_states`.

            # Run the appropriate decoder
            if target_charseqs is not None:
                # TODO: Create a self.DecoderTraining by passing `self` to its constructor.
                # Then run it on `[source_states, target_charseqs]` input,
                # storing the first result in `output_layer` and the third result in `output_lens`.
                pass
            else:
                # TODO: Create a self.DecoderPrediction by using:
                # - `self` as first argument to its constructor
                # - `maximum_iterations=tf.shape(source_charseqs)[1] + 10` as
                #   another argument, which indicates that the longest prediction
                #   must be at most 10 characters longer than the longest input.
                # Then run it on `source_states`, storing the first result
                # in `output_layer` and the third result in `output_lens`.
                pass

            # Reshape the output to the original matrix of lemmas
            # and explicitly set mask for loss and metric computation.
            output_layer = tf.scatter_nd(valid_words, output_layer, tf.concat([source_charseqs_shape[:2], tf.shape(output_layer)[1:]], axis=0))
            output_layer._keras_mask = tf.sequence_mask(tf.scatter_nd(valid_words, output_lens, source_charseqs_shape[:2]))
            return output_layer

    def __init__(self, args, num_source_chars, num_target_chars):
        self.lemmatizer = self.Lemmatizer(args, num_source_chars, num_target_chars)

        self.lemmatizer.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="character_accuracy")],
        )
        self.writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def append_eow(self, sequences):
        """Append EOW character after end of every given sequence."""
        padded_sequences = np.pad(sequences, [[0, 0], [0, 0], [0, 1]])
        ends = np.logical_xor(padded_sequences != 0, np.pad(sequences, [[0, 0], [0, 0], [1, 0]], constant_values=1) != 0)
        padded_sequences[ends] = MorphoDataset.Factor.EOW
        return padded_sequences

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            # TODO: Create `targets` by append EOW after target lemmas

            # TODO: Train the lemmatizer using `train_on_batch` method, storing
            # result metrics in `metrics`. You need to pass the `targets`
            # both on input and as gold labels.

            # Generate the summaries each 10 steps
            iteration = int(self.lemmatizer.optimizer.iterations)
            if iteration % 10 == 0:
                tf.summary.experimental.set_step(iteration)
                metrics = dict(zip(self.lemmatizer.metrics_names, metrics))

                predictions = self.predict_batch(batch[dataset.FORMS].charseqs[:1]).numpy()
                form = "".join(dataset.data[dataset.FORMS].alphabet[i] for i in batch[dataset.FORMS].charseqs[0, 0] if i)
                gold_lemma = "".join(dataset.data[dataset.LEMMAS].alphabet[i] for i in targets[0, 0] if i)
                system_lemma = "".join(dataset.data[dataset.LEMMAS].alphabet[i] for i in predictions[0, 0] if i != MorphoDataset.Factor.EOW)
                status = ", ".join([*["{}={:.4f}".format(name, value) for name, value in metrics.items()],
                                    "{} {} {}".format(form, gold_lemma, system_lemma)])
                print("Step {}:".format(iteration), status)

                with self.writer.as_default():
                    for name, value in metrics.items():
                        tf.summary.scalar("train/{}".format(name), value)
                    tf.summary.text("train/prediction", status)

    @tf.function(experimental_relax_shapes=True)
    def predict_batch(self, charseqs):
        return self.lemmatizer(charseqs)

    def evaluate(self, dataset, dataset_name, args):
        correct_lemmas, total_lemmas = 0, 0
        for batch in dataset.batches(args.batch_size):
            predictions = self.predict_batch(batch[dataset.FORMS].charseqs).numpy()

            # Compute whole lemma accuracy
            targets = self.append_eow(batch[dataset.LEMMAS].charseqs)
            resized_predictions = np.concatenate([predictions, np.zeros_like(targets)], axis=2)[:, :, :targets.shape[2]]
            valid_lemmas = targets[:, :, 0] != MorphoDataset.Factor.EOW

            total_lemmas += np.sum(valid_lemmas)
            correct_lemmas += np.sum(valid_lemmas * np.all(targets == resized_predictions * (targets != 0), axis=2))

        metrics = {"lemma_accuracy": correct_lemmas / total_lemmas}
        with self.writer.as_default():
            tf.summary.experimental.set_step(self.lemmatizer.optimizer.iterations)
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
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
        print("{:.2f}".format(100 * metrics["lemma_accuracy"]), file=out_file)
