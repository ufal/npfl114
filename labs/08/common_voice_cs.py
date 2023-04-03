import os
import sys
from typing import Dict, List, Optional, Sequence, TextIO, Tuple
import urllib.request
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf


class CommonVoiceCs:
    MFCC_DIM: int = 13

    LETTERS: List[str] = [
        "[UNK]", " ", "a", "á", "ä", "b", "c", "č", "d", "ď", "e", "é", "è",
        "ě", "f", "g", "h", "i", "í", "ï", "j", "k", "l", "m", "n", "ň", "o",
        "ó", "ö", "p", "q", "r", "ř", "s", "š", "t", "ť", "u", "ú", "ů", "ü",
        "v", "w", "x", "y", "ý", "z", "ž",
    ]

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/datasets/"

    @staticmethod
    def parse(example: tf.Tensor) -> Dict[str, tf.Tensor]:
        example = tf.io.parse_single_example(example, {
            "mfccs": tf.io.VarLenFeature(tf.float32),
            "sentence": tf.io.FixedLenFeature([], tf.string)})
        example["mfccs"] = tf.reshape(tf.sparse.to_dense(example["mfccs"]), [-1, CommonVoiceCs.MFCC_DIM])
        return example

    def __init__(self) -> None:
        for dataset, size in [("train", 9_773), ("dev", 904), ("test", 3_240)]:
            path = "common_voice_cs.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
                os.rename("{}.tmp".format(path), path)

            setattr(self, dataset, tf.data.TFRecordDataset(path).map(CommonVoiceCs.parse).apply(
                tf.data.experimental.assert_cardinality(size)))

        self._letters_mapping = tf.keras.layers.StringLookup(vocabulary=self.LETTERS[1:])

    train: tf.data.Dataset
    dev: tf.data.Dataset
    test: tf.data.Dataset

    @property
    def letters_mapping(self) -> tf.keras.layers.StringLookup:
        return self._letters_mapping

    # Methods for generating mfccs.
    @staticmethod
    def wav_decode(wav: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        audio, sample_rate = tf.audio.decode_wav(wav, desired_channels=1)
        return audio[:, 0], sample_rate

    @staticmethod
    def mfcc_extract(audio: tf.Tensor, sample_rate: int = 16000) -> tf.Tensor:
        assert sample_rate == 16000, "Only 16k sample rate is supported"

        # A 1024-point STFT with frames of 64 ms and 75% overlap.
        stfts = tf.signal.stft(audio, frame_length=1024, frame_step=256, fft_length=1024)
        spectrograms = tf.math.square(tf.math.abs(stfts))

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins, lower_edge_hertz, upper_edge_hertz, num_mel_bins = 513, 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.linalg.matmul(spectrograms, linear_to_mel_weight_matrix)

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first `CommonVoiceCs.MFCC_DIM`s.
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[:, :CommonVoiceCs.MFCC_DIM]

        return mfccs

    # Edit distance computation as Keras metric
    class EditDistanceMetric(tf.metrics.Mean):
        def __init__(self, name: str = "edit_distance", dtype: Optional[tf.DType] = None) -> None:
            super().__init__(name, dtype)

        def update_state(
                self, y_true: tf.RaggedTensor, y_pred: tf.RaggedTensor, sample_weight: Optional[tf.Tensor] = None
        ) -> None:
            """Computes edit distance for two RaggedTensors"""
            assert isinstance(y_true, tf.RaggedTensor) and isinstance(y_pred, tf.RaggedTensor)

            edit_distances = tf.edit_distance(y_pred.to_sparse(), y_true.to_sparse(), normalize=True)
            return super().update_state(edit_distances, sample_weight)

    # Evaluation infrastructure
    @staticmethod
    def evaluate(gold_dataset: tf.data.Dataset, predictions: Sequence[str]):
        gold = [np.array(example["sentence"]).item().decode("utf-8") for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        edit_distance = CommonVoiceCs.EditDistanceMetric()
        for i in range(0, len(gold), 16):
            edit_distance(
                tf.ragged.constant([list(sentence) for sentence in gold[i:i + 16]], tf.string),
                tf.ragged.constant([list(sentence) for sentence in predictions[i:i + 16]], tf.string),
            )

        return 100 * edit_distance.result()

    @staticmethod
    def evaluate_file(gold_dataset: tf.data.Dataset, predictions_file: TextIO) -> float:
        predictions = []
        for line in predictions_file:
            predictions.append(line.rstrip("\n"))
        return CommonVoiceCs.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            edit_distance = CommonVoiceCs.evaluate_file(getattr(CommonVoiceCs(), args.dataset), predictions_file)
        print("CommonVoiceCs edit distance: {:.2f}%".format(edit_distance))
