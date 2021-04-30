### Assignment: speech_recognition
#### Date: Deadline: May 10, 23:59
#### Points: 5 points+5 bonus

This assignment is a competition task in speech recognition area. Specifically,
your goal is to predict a sequence of letters given a spoken utterance.
We will be using Czech recordings from the [Common Voice](https://commonvoice.mozilla.org/),
with input sound waves passed through the usual preprocessing – computing
[Mel-frequency cepstral coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).
You can repeat this preprocessing on a given audio using the `wav_decode` and
`mfcc_extract` methods from the
[common_voice_cs.py](https://github.com/ufal/npfl114/tree/master/labs/09/common_voice_cs.py) module.
This module can also load the dataset, downloading it when necessary (note that
it has 200MB, so it might take a while). Furthermore, you can listen to the
[development portion of the dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/demos/common_voice_cs/).

This is an _open-data task_, where you submit only the test set annotations
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl114/2021-summer#competitions).
The evaluation is performed by computing the edit distance to the gold letter
sequence, normalized by its length (a corresponding Keras metric
`EditDistanceMetric` is provided by the [common_voice_cs.py](https://github.com/ufal/npfl114/tree/master/labs/09/common_voice_cs.py)).
Everyone who submits a solution with at most 50% test set edit distance
gets 5 points; the rest 5 points will be distributed
depending on relative ordering of your solutions. Note that
you can evaluate the predictions as usual using the [common_voice_cs.py](https://github.com/ufal/npfl114/tree/master/labs/09/common_voice_cs.py)
module, either by running with `--evaluate=path` arguments, or using its
`evaluate_file` method.

Start with the [speech_recognition.py](https://github.com/ufal/npfl114/tree/master/labs/09/speech_recognition.py)
template which contains instructions for using the CTC loss and generates the
test set annotation in the required format.
