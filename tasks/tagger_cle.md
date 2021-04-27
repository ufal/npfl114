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
Epoch 1/5 loss: 2.0234 - accuracy: 0.4140 - val_loss: 0.7620 - val_accuracy: 0.7860
Epoch 2/5 loss: 0.4854 - accuracy: 0.9083 - val_loss: 0.3223 - val_accuracy: 0.9193
Epoch 3/5 loss: 0.1271 - accuracy: 0.9840 - val_loss: 0.2438 - val_accuracy: 0.9299
Epoch 4/5 loss: 0.0666 - accuracy: 0.9891 - val_loss: 0.2243 - val_accuracy: 0.9302
Epoch 5/5 loss: 0.0486 - accuracy: 0.9902 - val_loss: 0.2200 - val_accuracy: 0.9281
loss: 0.2382 - accuracy: 0.9253
```
- `python3 tagger_cle.py --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16 --word_masking=0.1`
```
Epoch 1/5 loss: 2.0566 - accuracy: 0.4027 - val_loss: 0.8460 - val_accuracy: 0.7256
Epoch 2/5 loss: 0.6681 - accuracy: 0.8307 - val_loss: 0.3583 - val_accuracy: 0.9113
Epoch 3/5 loss: 0.2556 - accuracy: 0.9468 - val_loss: 0.2492 - val_accuracy: 0.9273
Epoch 4/5 loss: 0.1523 - accuracy: 0.9649 - val_loss: 0.2223 - val_accuracy: 0.9319
Epoch 5/5 loss: 0.1204 - accuracy: 0.9688 - val_loss: 0.2031 - val_accuracy: 0.9351
loss: 0.2258 - accuracy: 0.9289
```
#### Examples End:
