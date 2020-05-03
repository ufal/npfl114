### Assignment: tagger_cle_rnn
#### Date: Deadline: May 03, 23:59
#### Points: 2 points
#### Examples: tagger_cle_rnn_example

This assignment is a continuation of `tagger_we`. Using the
[tagger_cle_rnn.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_cle_rnn.py)
template, implement character-level word embedding computation using
a bidirectional character-level GRU.

Once submitted to ReCodEx, you should experiment with the effect of CLEs
compared to a plain `tagger_we`, and the influence of their dimensionality. Note
that `tagger_we` has by default word embeddings twice the size of word
embeddings in `tagger_cle_rnn`.

#### Examples Start: tagger_cle_rnn_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 tagger_cle_rnn.py --recodex --seed=7 --batch_size=3 --epochs=2 --threads=1 --max_sentences=90 --rnn_cell=LSTM --rnn_cell_dim=16 --we_dim=32 --cle_dim=16`
  ```
  25.85
  ```
- `python3 tagger_cle_rnn.py --recodex --seed=7 --batch_size=3 --epochs=2 --threads=1 --max_sentences=90 --rnn_cell=GRU --rnn_cell_dim=20 --we_dim=32 --cle_dim=16`
  ```
  33.90
  ```
#### Examples End:
