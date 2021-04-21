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
- `python3 tagger_cle.py --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16`
```
Epoch 1/5 loss: 1.9636 - accuracy: 0.4266 - val_loss: 0.8544 - val_accuracy: 0.7826
Epoch 2/5 loss: 0.6335 - accuracy: 0.8495 - val_loss: 0.3805 - val_accuracy: 0.9105
Epoch 3/5 loss: 0.1855 - accuracy: 0.9768 - val_loss: 0.2569 - val_accuracy: 0.9274
Epoch 4/5 loss: 0.0816 - accuracy: 0.9898 - val_loss: 0.2314 - val_accuracy: 0.9282
Epoch 5/5 loss: 0.0510 - accuracy: 0.9920 - val_loss: 0.2081 - val_accuracy: 0.9333
loss: 0.2254 - accuracy: 0.9356
```
- `python3 tagger_cle.py --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16 --word_masking=0.1`
```
Epoch 1/5 loss: 1.9811 - accuracy: 0.4198 - val_loss: 0.9202 - val_accuracy: 0.7548
Epoch 2/5 loss: 0.7492 - accuracy: 0.7979 - val_loss: 0.4502 - val_accuracy: 0.8995
Epoch 3/5 loss: 0.3066 - accuracy: 0.9406 - val_loss: 0.2839 - val_accuracy: 0.9311
Epoch 4/5 loss: 0.1634 - accuracy: 0.9663 - val_loss: 0.2291 - val_accuracy: 0.9291
Epoch 5/5 loss: 0.1157 - accuracy: 0.9725 - val_loss: 0.2195 - val_accuracy: 0.9308
loss: 0.2430 - accuracy: 0.9247
```
#### Examples End:
