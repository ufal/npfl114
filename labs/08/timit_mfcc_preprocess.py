#!/usr/bin/env python3

import tensorflow as tf

def wav_decode(wav):
    audio, sample_rate = tf.audio.decode_wav(wav, desired_channels=1)
    return audio[:, 0].numpy(), sample_rate.numpy()

def mfcc_extract(audio, sample_rate=16000):
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

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[:, :13]

    # Create deltas for MFCCs.
    mfccs = tf.concat([mfccs[:1], mfccs, mfccs[-1:]], axis=0)
    mfccs = tf.concat([mfccs[1:-1], mfccs[2:] - mfccs[:-2]], axis=1)

    return mfccs.numpy()
