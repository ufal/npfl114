This task is a continuation of `tagger_we` assignment.

Using the [tagger_cnne.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_cnne.py)
template, add the following features in addition to `tagger_we` ones:
- Create character embeddings for training alphabet.
- Process unique words with one-dimensional convolutional filters with
  kernel size of 2 to some given maximum. To obtain a fixed-size representation,
  perform chanel-wise max-pooling over the whole word.
- Generate convolutional embeddings (CNNE) as a concatenation of features
  corresponding to the ascending kernel sizes.
- Properly distribute the CNNEs of unique words into the batches of sentences.
- Generate overall embeddings by concatenating word-level embeddings and CNNEs.

Once submitted to ReCodEx, you should experiment with the effect of CNNEs
compared to plain `tagger_we`, and the influence of the maximum kernel size and
number of filters. Note that `tagger_we` has by default word embeddings twice
the size of word embeddings in `tagger_cnne`.
