import os
import sys
import urllib.request
import zipfile

import numpy as np

# Note: Because images have different size, the user
# - can specify `image_processing` method to dataset construction, which
#   is applied to every image during loading;
# - and/or can specify `image_processing` method to `batches` call, which is
#   applied to an image during batch construction.
#
# In any way, the batch images must be Numpy arrays with shape (224, 224, 3)
# and type np.float32. (In order to convert tf.Tensor to Numpty array
# use `tf.Tensor.numpy()` method.)
#
# If all images are of the above datatype after dataset construction
# (i.e., `image_processing` passed to `Caltech42` already generates such images),
# then `data["images"]` is a Numpy array with the images. Otherwise, it is
# a Python list of images, and the Numpy array is constructed only in `batches` call.

class Caltech42:
    labels = [
        "airplanes", "bonsai", "brain", "buddha", "butterfly",
        "car_side", "chair", "chandelier", "cougar_face", "crab",
        "crayfish", "dalmatian", "dragonfly", "elephant", "ewer",
        "faces", "flamingo", "grand_piano", "hawksbill", "helicopter",
        "ibis", "joshua_tree", "kangaroo", "ketch", "lamp", "laptop",
        "llama", "lotus", "menorah", "minaret", "motorbikes", "schooner",
        "scorpion", "soccer_ball", "starfish", "stop_sign", "sunflower",
        "trilobite", "umbrella", "watch", "wheelchair", "yin_yang",
    ]
    MIN_SIZE, C, LABELS = 224, 3, len(labels)

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/caltech42.zip"

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

        def batches(self, size=None, image_processing=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    if key == "images":
                        batch[key] = np.zeros([batch_size, Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C], dtype=np.float32)
                        for i, index in enumerate(batch_perm):
                            data = image_processing(self._data[key][index]) if image_processing is not None else self._data[key][index]
                            if type(data) != np.ndarray:
                                raise ValueError("Caltech42: Expecting images after `image_processing` to be Numpy `ndarray`")
                            if data.dtype != np.float32 or data.shape != (Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C):
                                raise ValueError("Caltech42: Expecting images after `image_processing` to have shape {} and dtype {}".format(
                                    (Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C), np.float32))
                            batch[key][i] = image_processing(self._data[key][index]) if image_processing is not None else self._data[key][index]
                    else:
                        batch[key] = self._data[key][batch_perm]
                yield batch

    def __init__(self, image_processing=None):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading Caltech42 dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)

        with zipfile.ZipFile(path, "r") as caltech42_zip:
            for dataset in ["train", "dev", "test"]:
                data = {"images": [], "labels": []}
                for name in sorted(caltech42_zip.namelist()):
                    if not name.startswith(dataset) or not name.endswith(".jpg"): continue

                    with caltech42_zip.open(name, "r") as image_file:
                        data["images"].append(image_file.read())
                        if image_processing is not None:
                            data["images"][-1] = image_processing(data["images"][-1])

                    if "_" in name:
                        data["labels"].append(self.labels.index(name[name.index("_")+1:-4]))
                    else:
                        data["labels"].append(-1)

                if all(map(lambda i: type(i) == np.ndarray and i.dtype == np.float32 and i.shape == (self.MIN_SIZE, self.MIN_SIZE, self.C),
                           data["images"])):
                    data["images"] = np.array(data["images"])
                data["labels"] = np.array(data["labels"], dtype=np.uint8)
                setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train"))
