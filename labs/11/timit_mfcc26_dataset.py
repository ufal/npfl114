#!/usr/bin/env python3
import numpy as np

class TIMIT:
    """ The class for loading TIMIT dataset with preprocessed MFCC coefficients.

    The class has the following properties:
    - phones: list of phonemes (strings)
    - mfcc_dim: number of mfcc coefficients in the preprocessed data
    - train: training portion
    - dev: dev portion
    - test: test portion

    The training/dev/test portions are objects with the following methods:
    - next_batch(batch_size): returns a quadruple (mfcc_lens, mfccs, phone_lens, phones)
    - epoch_finished(): True/False if the current epoch is finished
    """

    class Dataset:
        def __init__(self, data, shuffle_batches):
            self._mfccs = data["mfcc"]
            self._phones = data["phones"]
            self._mfcc_dim = self._mfccs[0].shape[1]

            self._mfcc_lens = np.zeros([len(self._mfccs)], np.int32)
            for i in range(len(self._mfccs)):
                self._mfcc_lens[i] = len(self._mfccs[i])

            self._phone_lens = np.zeros([len(self._phones)], np.int32)
            for i in range(len(self._phones)):
                self._phone_lens[i] = len(self._phones[i])

            self._shuffle_batches = shuffle_batches
            self._permutation = np.random.permutation(len(self._mfccs)) if self._shuffle_batches else np.arange(len(self._mfccs))

        def next_batch(self, batch_size):
            batch_size = min(batch_size, len(self._permutation))
            batch_perm = self._permutation[:batch_size]
            self._permutation = self._permutation[batch_size:]

            mfcc_lens = self._mfcc_lens[batch_perm]
            mfccs = np.zeros([batch_size, np.max(mfcc_lens), self._mfcc_dim])
            for i in range(batch_size): mfccs[i, :mfcc_lens[i]] = self._mfccs[batch_perm[i]]

            phone_lens = self._phone_lens[batch_perm]
            phones = np.zeros([batch_size, np.max(phone_lens)], np.int64)
            for i in range(batch_size): phones[i, :phone_lens[i]] = self._phones[batch_perm[i]]

            return mfcc_lens, mfccs, phone_lens, phones

        def epoch_finished(self):
            if len(self._permutation) == 0:
                self._permutation = np.random.permutation(len(self._mfccs)) if self._shuffle_batches else np.arange(len(self._mfccs))
                return True
            return False

    def __init__(self, filename):
        import pickle

        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        self._phones = data["phonemes"]

        self._train = self.Dataset(data["train"], shuffle_batches=True)
        self._dev = self.Dataset(data["dev"], shuffle_batches=False)
        self._test = self.Dataset(data["test"], shuffle_batches=False)

    @property
    def phones(self):
        return self._phones

    @property
    def mfcc_dim(self):
        return self._train._mfcc_dim

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test
