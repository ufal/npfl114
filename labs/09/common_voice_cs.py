import os
import sys
import urllib.request
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import tensorflow as tf

class CommonVoiceCs:
    LETTERS = [
        "[UNK]", " ", "a", "á", "ä", "b", "c", "č", "d", "ď", "e", "é", "è",
        "ě", "f", "g", "h", "i", "í", "ï", "j", "k", "l", "m", "n", "ň", "o",
        "ó", "ö", "p", "q", "r", "ř", "s", "š", "t", "ť", "u", "ú", "ů", "ü",
        "v", "w", "x", "y", "ý", "z", "ž",
    ]

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/datasets/"

    @staticmethod
    def parse(example):
        example = tf.io.parse_single_example(example, {
            "mfccs": tf.io.VarLenFeature(tf.float32),
            "sentence": tf.io.FixedLenFeature([], tf.string)})
        example["mfccs"] = tf.reshape(tf.cast(tf.sparse.to_dense(example["mfccs"]), tf.float32), [-1, 13])
        return example

    def __init__(self):
        for dataset, size in [("train", 9773), ("dev", 904), ("test", 3240)]:
            path = "common_voice_cs.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

            setattr(self, dataset,
                    tf.data.TFRecordDataset(path).map(CommonVoiceCs.parse).apply(tf.data.experimental.assert_cardinality(size)))

        self._letters_mapping = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
        self._letters_mapping.set_vocabulary(self.LETTERS[1:])

    @property
    def letters_mapping(self):
        return self._letters_mapping

    class EditDistanceMetric(tf.metrics.Mean):
        def update_state(self, y_true, y_pred, sample_weight=None):
            """Computes edit distance for two RaggedTensors"""

            edit_distances = tf.edit_distance(y_pred.to_sparse(), y_true.to_sparse(), normalize=True)
            return super().update_state(edit_distances, sample_weight)

    # Evaluation infrastructure will be finished later.
