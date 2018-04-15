This task is a continuation of `tagger_we` assignment.

Using the [tagger_cle.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_cle.py)
template, add the following features in addition to `tagger_we` ones:
- Create character embeddings for training alphabet.
- Process unique words with a bidirectional character-level RNN.
- Create character word-level embeddings as a sum of the final forward and
  backward state.
- Properly distribute the CLEs of unique words into the batches of sentences.
- Generate overall embeddings by concatenating word-level embeddings and CLEs.

Once submitted to ReCodEx, you should experiment with the effect of CLEs
compared to plain `tagger_we`, and the influence of their dimensionality.
Note that `tagger_we` has by default word embeddings twice the size of
word embeddings in `tagger_cle`.
