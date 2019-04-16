### Assignment: tagger_cle_rnn
#### Date: Deadline: Apr 28, 23:59
#### Points: 3 points

This task is a continuation of [`tagger_we`](#tagger_we) assignment. Using the
[tagger_cle_rnn.py](https://github.com/ufal/npfl114/tree/master/labs/07/tagger_cle_rnn.py)
template, implement the following features in addition to [`tagger_we`](#tagger_we):
- Create character embeddings for training alphabet.
- Process unique words with a bidirectional character-level RNN, concatenating
  the results.
- Properly distribute the CLEs of unique words into the batches of sentences.
- Generate overall embeddings by concatenating word-level embeddings and CLEs.

Once submitted to ReCodEx, continue with [`tagger_cle_cnn`](#tagger_cle_cnn)
assignment. Additionaly, you should experiment with the effect of CLEs compared
to plain [`tagger_we`](#tagger_we), and the influence of their dimensionality.
Note that [`tagger_we`](#tagger_we) has by default word embeddings twice the
size of word embeddings in [`tagger_cle_rnn`](#tagger_cle_rnn).
