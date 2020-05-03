### Assignment: tagger_cle_cnn
#### Date: Deadline: May 03, 23:59
#### Points: 2 points
#### Examples: tagger_cle_cnn_example

This task is a continuation of `tagger_cle_rnn` assignment. Using the
[tagger_cle_cnn.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_cle_cnn.py)
template, instead of using RNNs to generate character-level embeddings,
process character sequences with 1D convolutional filters with varying kernel
sizes and obtain fixed-size representations using global max-pooling.
Compute the final embeddings by using a highway layer.

#### Examples Start: tagger_cle_cnn_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 tagger_cle_cnn.py --recodex --seed=7 --batch_size=3 --epochs=4 --threads=1 --max_sentences=90 --rnn_cell=LSTM --rnn_cell_dim=16 --we_dim=32 --cle_dim=16 --cnn_filters=16 --cnn_max_width=3`
  ```
  38.01
  ```
- `python3 tagger_cle_cnn.py --recodex --seed=7 --batch_size=3 --epochs=4 --threads=1 --max_sentences=90 --rnn_cell=GRU --rnn_cell_dim=20 --we_dim=32 --cle_dim=16 --cnn_filters=16 --cnn_max_width=3`
  ```
  53.85
  ```
#### Examples End:
