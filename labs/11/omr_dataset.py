import os
import pickle
import sys
import urllib.request

import numpy as np

# Note: Because images have different size, the user
# - can specify `image_processing` method to dataset construction, which
#   is applied to every image during loading;
# - and/or can specify `image_processing` method to `batches` call, which is
#   applied to an image during batch construction.
#
# In any way, the batch images must be Numpy arrays with type np.float32.
# The images in the batch are padded with (-1) in place of invalid pixels.
# (In order to convert tf.Tensor to Numpy array use `tf.Tensor.numpy()` method.)
#
# The target marks in a batch are a Numpy array with np.uint16 type, padded
# using zeros.

class OMRDataset:
    # The MARKS field is defined on the loaded instance of the dataset and contains
    # a list of names of the target marks to recognize.
    MARKS = None

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/omr_dataset.pickle"

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._size = len(self._data["images"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None, image_processing=lambda x: x):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                # Process examples
                data = {"images": [], "marks": []}
                for i in batch_perm:
                    image = image_processing(self._data["images"][i])
                    if type(image) != np.ndarray:
                        raise ValueError("OMR_dataset: Expecting images after `image_processing` to be Numpy `ndarray`")
                    if image.dtype != np.float32:
                        raise ValueError("OMR_dataset: Expecting images after `image_processing` to have dtype `np.float32`")
                    if len(image.shape) != 3:
                        raise ValueError("OMR_dataset: Expecting images after `image_processing` to have 3 dimensions")

                    data["images"].append(image)
                    data["marks"].append(self._data["marks"][i])

                # Compute maximum sizes
                max_h, max_w, max_c = [max(image.shape[d] for image in data["images"]) for d in range(3)]
                max_targets = max(marks.shape[0] for marks in data["marks"])

                batch = {
                    "images": np.full([batch_size, max_h, max_w, max_c], -1, dtype=np.float32),
                    "marks": np.zeros([batch_size, max_targets], dtype=np.uint16),
                }
                for i in range(batch_size):
                    image, mark = data["images"][i], data["marks"][i]
                    batch["images"][i, :image.shape[0], :image.shape[1], :image.shape[2]] = image
                    batch["marks"][i, :mark.shape[0]] = mark

                yield batch

    def __init__(self, image_processing=lambda x: x):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading OMR dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)

        with open(path, "rb") as omr_file:
            data = pickle.load(omr_file)

        self.MARKS = data["marks"]

        for dataset in ["train", "dev", "test"]:
            processed = {"images": [], "marks": []}

            for i in range(len(data[dataset]["images"])):
                processed["images"].append(image_processing(data[dataset]["images"][i]))
                processed["marks"].append(np.array(data[dataset]["marks"][i], dtype=np.uint16))

            setattr(self, dataset, self.Dataset(processed, shuffle_batches=dataset == "train"))
