### Assignment: tagger_cle
#### Date: Deadline: May 03, 23:59
#### Points: 3 points
#### Examples: tagger_cle_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_cle.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_cle.py)
template, implement character-level word embedding computation using
a bidirectional character-level GRU.

Once submitted to ReCodEx, you should experiment with the effect of CLEs
compared to a plain `tagger_we`, and the influence of their dimensionality. Note
that `tagger_cle` has by default smaller word embeddings so that the size
of word representation (64 + 32 + 32) is the same as in the `tagger_we` assignment.

#### Examples Start: tagger_cle_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 tagger_cle.py --max_sentences=5000 --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16`
```
Epoch 1/5 loss: 1.8425 - accuracy: 0.4607 - val_loss: 0.4031 - val_accuracy: 0.9008
Epoch 2/5 loss: 0.2080 - accuracy: 0.9599 - val_loss: 0.2516 - val_accuracy: 0.9204
Epoch 3/5 loss: 0.0560 - accuracy: 0.9882 - val_loss: 0.2177 - val_accuracy: 0.9286
Epoch 4/5 loss: 0.0335 - accuracy: 0.9917 - val_loss: 0.2155 - val_accuracy: 0.9265
Epoch 5/5 loss: 0.0250 - accuracy: 0.9935 - val_loss: 0.1920 - val_accuracy: 0.9363
loss: 0.2118 - accuracy: 0.9289
```
- `python3 tagger_cle.py --max_sentences=5000 --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16 --word_masking=0.1`
```
Epoch 1/5 loss: 1.8989 - accuracy: 0.4426 - val_loss: 0.4616 - val_accuracy: 0.8798
Epoch 2/5 loss: 0.3442 - accuracy: 0.9155 - val_loss: 0.2408 - val_accuracy: 0.9265
Epoch 3/5 loss: 0.1503 - accuracy: 0.9605 - val_loss: 0.1994 - val_accuracy: 0.9364
Epoch 4/5 loss: 0.1040 - accuracy: 0.9706 - val_loss: 0.1847 - val_accuracy: 0.9427
Epoch 5/5 loss: 0.0892 - accuracy: 0.9728 - val_loss: 0.1882 - val_accuracy: 0.9401
loss: 0.2029 - accuracy: 0.9361
```
#### Examples End:
