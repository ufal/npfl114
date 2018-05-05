#!/usr/bin/env python3
import wave

import numpy as np

# Version 0.6 was used; https://github.com/jameslyons/python_speech_features
import python_speech_features

_mean = np.array([
    1.33700586e+01, -1.03433741e+01, -1.15338359e+01, -6.46144121e+00,
    -1.58748411e+01, -9.75004999e+00, -7.84680256e+00, -6.91626105e+00,
    -1.56269805e-01, -4.43677753e+00, -1.29033844e+00, -3.28106845e+00,
    -3.27096609e+00, -6.34995403e-04, 9.47158201e-03, 1.28413704e-02,
    5.46076000e-03, 4.07909218e-03, 1.24261320e-02, 1.95510662e-03,
    2.44855866e-03, -8.50233425e-03, -4.96037454e-03, 3.69057617e-03,
    -3.60272821e-03, -2.82779786e-03
])

_std = np.array([
    3.32128623, 18.24345103, 14.45241678, 15.97572181, 15.71888497,
    16.19171175, 15.6327044, 15.29295722, 14.60782802, 13.8446527,
    12.10992735, 11.76798047, 10.60377405, 0.90783704, 4.92753412,
    4.85200799, 4.59985813, 5.23554332, 5.29150059, 5.21931908,
    5.22870299, 5.34311608, 5.03691209, 4.68988606, 4.42778807,
    4.09015497
])

def process_wav(wav_file, normalize=True):
    with wave.open(wav_file, mode="rb") as wav:
        if wav.getcomptype() != "NONE":
            raise RuntimeError("Unsupported compression {}".format(wav.getcomptype()))

        if wav.getnchannels() != 1:
            raise RuntimeError("Only mono WAVs are supported")

        if wav.getsampwidth() != 2:
            raise RuntimeError("Only 16bit WAVs are supported")

        rate = wav.getframerate()
        data = np.fromstring(wav.readframes(-1), "<i2")

    mfcc = python_speech_features.mfcc(data, rate, winlen=0.025, winstep=0.01, numcep=13, appendEnergy=True)
    delta = python_speech_features.delta(mfcc, 1)
    joined = np.concatenate([mfcc, delta], axis=1)

    if normalize:
        joined = (joined - _mean) / _std

    return joined
