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
Epoch 1/5 loss: 2.0681 - accuracy: 0.4102 - val_loss: 0.9439 - val_accuracy: 0.7096
Epoch 2/5 loss: 0.6816 - accuracy: 0.8201 - val_loss: 0.4189 - val_accuracy: 0.8889
Epoch 3/5 loss: 0.1758 - accuracy: 0.9787 - val_loss: 0.2937 - val_accuracy: 0.9077
Epoch 4/5 loss: 0.0726 - accuracy: 0.9911 - val_loss: 0.2656 - val_accuracy: 0.9082
Epoch 5/5 loss: 0.0465 - accuracy: 0.9927 - val_loss: 0.2594 - val_accuracy: 0.9099
loss: 0.2837 - accuracy: 0.9028
```
- `python3 tagger_cle.py --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16 --word_masking=0.1`
```
Epoch 1/5 loss: 2.1060 - accuracy: 0.4001 - val_loss: 1.0534 - val_accuracy: 0.6603
Epoch 2/5 loss: 0.8794 - accuracy: 0.7332 - val_loss: 0.5216 - val_accuracy: 0.8884
Epoch 3/5 loss: 0.3569 - accuracy: 0.9331 - val_loss: 0.3260 - val_accuracy: 0.9162
Epoch 4/5 loss: 0.1914 - accuracy: 0.9601 - val_loss: 0.2659 - val_accuracy: 0.9236
Epoch 5/5 loss: 0.1416 - accuracy: 0.9658 - val_loss: 0.2491 - val_accuracy: 0.9266
loss: 0.2632 - accuracy: 0.9258
```
#### Examples End:
