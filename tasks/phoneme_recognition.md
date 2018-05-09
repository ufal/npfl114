This assignment is a competition task in speech recognition area. Specifically,
your goal is to predict a sequence of phonemes given a spoken utterance.
We will be using TIMIT corpus, with input sound waves passed through the usual
preprocessing – computing 13
[Mel-frequency cepstral coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
each 10 milliseconds and appending their derivation, obtaining 26 floats for
every 10 milliseconds. You can repeat exactly this preprocessing on a given `wav`
file using the [timit_mfcc26_preprocess.py](https://github.com/ufal/npfl114/tree/master/labs/11/timit_mfcc26_preprocess.py)
script.

Because the data is not publicly available, you can download it only through
ReCodEx. Please do not distribute it. To load the dataset, you can use
[timit_mfcc26_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/11/timit_mfcc26_dataset.py)
module.

This task is an open-data competition and the points will be awarded depending on your
test set performance. The generated phoneme sequences are evaluated using edit distance to the gold
phoneme sequence, normalized by the length of the phoneme sequence
(i.e., exactly as `tf.edit_distance`). If your test set score surpasses 50%, you will be
awarded 6 points; the rest 6 points will be distributed depending on relative
ordering of your solutions. An evaluation (using for example development data)
can be performed by [timit_mfcc26_eval.py](https://github.com/ufal/npfl114/tree/master/labs/11/timit_mfcc26_eval.py).

You can start with the
[phoneme_recognition.py](https://github.com/ufal/npfl114/tree/master/labs/11/phoneme_recognition.py)
template. You will need to implement the following:
- The CTC loss and CTC decoder employ sparse tensor – therefore, start by
  [studying them](https://www.tensorflow.org/api_guides/python/sparse_ops).
- Convert the input phoneme sequences into sparse representation
  (`tf.where` and `tf.gather_nd` are useful).
- Use a bidirectional RNN and an output linear layer without activation.
- Utilize CTC loss (`tf.nn.ctc_loss`).
- Perform decoding by a CTC decoder (either greedily using
  `tf.nn.ctc_greedy_decoder`, or with beam search employing
  `tf.nn.ctc_beam_search_decoder`).
- Evaluate results using normalized edit distance (`tf.edit_distance`).
- Write the generated phoneme sequences.

To submit the test set annotations in ReCodEx, use the supplied
[phoneme_recognition_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/11/phoneme_recognition_recodex.py)
script. You need to provide at least two arguments – the first is the path to
the test set annotations and all other arguments are paths to the sources used
to generate the test data.
