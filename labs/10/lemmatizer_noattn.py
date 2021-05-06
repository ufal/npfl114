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
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Network(tf.keras.Model):
    def __init__(self, args, train):
        super().__init__()

        self.source_mapping = train.forms.char_mapping
        self.target_mapping = train.lemmas.char_mapping
        self.target_mapping_inverse = type(self.target_mapping)(
            mask_token=None, vocabulary=self.target_mapping.get_vocabulary(), invert=True)

        # TODO: Define
        # - `self.source_embedding` as an embedding layer of source chars into `args.cle_dim` dimensions
        # - `self.source_rnn` as a bidirectional GRU with `args.rnn_dim` units, returning only the last output,
        #   summing opposite directions

        # TODO: Then define
        # - `self.target_embedding` as an embedding layer of target chars into `args.cle_dim` dimensions
        # - `self.target_rnn_cell` as a GRUCell with `args.rnn_dim` units
        # - `self.target_output_layer` as a Dense layer into as many outputs as there are unique target chars

        # Compile the model
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.Accuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

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
            # embedded using self.lemmatizer.target_embedding

            # TODO: Define `states` as self.source_states

            return finished, inputs, states

        def step(self, time, inputs, states, training):
            # TODO: Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.

            # TODO: Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,

            # TODO: Define `next_inputs` by embedding `time`-th chars from `self.targets`.

            # TODO: Define `finished` as True if `time`-th char from `self.targets` is
            # `MorphoDataset.Factor.EOW`, False otherwise.

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
            # Use `initialize` from the DecoderTraining, passing None as targets
            return super().initialize([layer_inputs, None], initial_state)

        def step(self, time, inputs, states, training):
            # TODO(DecoderTraining): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.

            # TODO(DecoderTraining): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,

            # TODO: Overwrite `outputs` by passing them through `tf.argmax` on suitable axis and with
            # `output_type=tf.int32` parameter.

            # TODO: Define `next_inputs` by embedding the `outputs`

            # TODO: Define `finished` as True if `outputs are `MorphoDataset.Factor.EOW`, False otherwise.

            return outputs, states, next_inputs, finished

    # If `targets` is given, we are in the teacher forcing mode.
    # Otherwise, we run in autoregressive mode.
    def call(self, inputs, targets=None):
        # FIX: Get indices of valid lemmas and reshape the `source_charseqs`
        # so that it is a list of valid sequences, instead of a
        # matrix of sequences, some of them padding ones.
        source_charseqs = inputs.values
        source_charseqs = tf.strings.unicode_split(source_charseqs, "UTF-8")
        source_charseqs = self.source_mapping(source_charseqs)
        if targets is not None:
            target_charseqs = targets.values
            target_charseqs = target_charseqs.to_tensor()

        # TODO: Embed source_charseqs using `source_embedding`

        # TODO: Run source_rnn on the embedded sequences, returning outputs in `source_states`.

        # Run the appropriate decoder
        if targets is not None:
            # TODO: Create a self.DecoderTraining by passing `self` to its constructor.
            # Then run it on `[source_states, target_charseqs]` input,
            # storing the first result in `output` and the third result in `output_lens`.
            raise NotImplementedError()
        else:
            # TODO: Create a self.DecoderPrediction by using:
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
            [y_targets, tf.fill([y_targets.bounding_shape(0), 1], tf.constant(MorphoDataset.Factor.EOW, tf.int64))], axis=-1)
        y_targets = y.with_values(y_targets)

        with tf.GradientTape() as tape:
            y_pred = self(x, targets=y_targets, training=True)
            loss = self.compiled_loss(y_targets.flat_values, y_pred.flat_values, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    def predict_step(self, data):
        if isinstance(data, tuple): data = data[0]
        y_pred = self(data, training=False)
        y_pred = self.target_mapping_inverse(y_pred)
        y_pred = tf.strings.reduce_join(y_pred, axis=-1)
        return y_pred

    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(data)
        self.compiled_metrics.update_state(tf.ones_like(y.values, dtype=tf.int32), tf.cast(y_pred.values == y.values, tf.int32))
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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences, add_bow_eow=True)

    # Create the network and train
    network = Network(args, morpho.train)

    # Construct dataset for lemmatizer training
    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(lambda forms, lemmas, tags: (forms, lemmas))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    # Callback showing intermediate results during training
    class ShowIntermediateResults(tf.keras.callbacks.Callback):
        def __init__(self, data):
            self._iterator = iter(data.repeat())
        def on_train_batch_end(self, batch, logs=None):
            if network.optimizer.iterations % 10 == 0:
                forms, lemmas = next(self._iterator)
                tf.print(network.optimizer.iterations, forms[0, 0], lemmas[0, 0], network.predict_on_batch(forms[:1, :1])[0, 0])

    network.fit(train, epochs=args.epochs, validation_data=dev, verbose=2,
                callbacks=[ShowIntermediateResults(dev), network.tb_callback])

    test_logs = network.evaluate(test, return_dict=True)
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    # Return test set accuracy for ReCodEx to validate
    return test_logs["accuracy"]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
