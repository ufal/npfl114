### Assignment: speech_recognition
#### Date: Deadline: Apr 28, 23:59
#### Points: 7-12 points

This assignment is a competition task in speech recognition area. Specifically,
your goal is to predict a sequence of letters given a spoken utterance.
We will be using TIMIT corpus, with input sound waves passed through the usual
preprocessing – computing
[Mel-frequency cepstral coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).
You can repeat exactly this preprocessing on a given audio
using the [timit_mfcc_preprocess.py](https://github.com/ufal/npfl114/tree/master/labs/07/timit_mfcc_preprocess.py)
script.

Because the data is not publicly available, you can download it only through
ReCodEx. Please do not distribute it. To load the dataset using the
[timit_mfcc.py](https://github.com/ufal/npfl114/tree/master/labs/07/timit_mfcc.py) module.

This is an _open-data task_, where you submit only the test set labels
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

The task is also a _competition_. The evaluation is performed by computing edit
distance to the gold letter sequence, normalized by its length (i.e., exactly as
`tf.edit_distance`). Everyone who submits a solution which achieves
at most _50%_ test set edit distance will get 7 points; the rest 5 points will be distributed
depending on relative ordering of your solutions. An evaluation (using for example development data)
can be performed by
[speech_recognition_eval.py](https://github.com/ufal/npfl114/tree/master/labs/07/speech_recognition_eval.py).

You should start with the
[speech_recognition.py](https://github.com/ufal/npfl114/tree/master/labs/07/speech_recognition.py)
template.
- To perform speech recognition, you should use CTC loss for training and CTC
  beam search decoder for prediction. Both the CTC loss and CTC decoder employ
  sparse tensor – therefore, start by
  [studying them](https://www.tensorflow.org/api_guides/python/sparse_ops).
- The basic architecture:
  - converts target letters into sparse representation,
  - use a bidirectional RNN and an output linear layer without activation,
  - compute CTC loss (`tf.nn.ctc_loss`),
  - if required, perform decoding by a CTC decoder (`tf.nn.ctc_beam_search_decoder`)
    and possibly evaluate results using normalized edit distance (`tf.edit_distance`).
