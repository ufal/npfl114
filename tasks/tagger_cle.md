### Assignment: tagger_cle
#### Date: Deadline: Apr 11, 7:59 a.m.
#### Points: 3 points
#### Examples: tagger_cle_examples
#### Tests: tagger_cle_tests

This assignment is a continuation of `tagger_we`. Using the
[tagger_cle.py](https://github.com/ufal/npfl114/tree/master/labs/07/tagger_cle.py)
template, implement character-level word embedding computation using
a bidirectional character-level GRU.

Once submitted to ReCodEx, you should experiment with the effect of CLEs
compared to a plain `tagger_we`, and the influence of their dimensionality. Note
that `tagger_cle` has by default smaller word embeddings so that the size
of word representation (64 + 32 + 32) is the same as in the `tagger_we` assignment.

#### Examples Start: tagger_cle_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_cle.py --max_sentences=5000 --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16`
```
Epoch 1/5 loss: 1.2229 - accuracy: 0.6372 - val_loss: 0.4645 - val_accuracy: 0.8702
Epoch 2/5 loss: 0.1907 - accuracy: 0.9598 - val_loss: 0.2491 - val_accuracy: 0.9249
Epoch 3/5 loss: 0.0557 - accuracy: 0.9883 - val_loss: 0.2151 - val_accuracy: 0.9267
Epoch 4/5 loss: 0.0344 - accuracy: 0.9910 - val_loss: 0.2125 - val_accuracy: 0.9277
Epoch 5/5 loss: 0.0262 - accuracy: 0.9925 - val_loss: 0.2069 - val_accuracy: 0.9295
```
- `python3 tagger_cle.py --max_sentences=5000 --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16 --word_masking=0.1`
```
Epoch 1/5 loss: 1.3114 - accuracy: 0.6076 - val_loss: 0.5267 - val_accuracy: 0.8527
Epoch 2/5 loss: 0.3150 - accuracy: 0.9197 - val_loss: 0.2760 - val_accuracy: 0.9161
Epoch 3/5 loss: 0.1540 - accuracy: 0.9588 - val_loss: 0.2244 - val_accuracy: 0.9294
Epoch 4/5 loss: 0.1123 - accuracy: 0.9676 - val_loss: 0.2145 - val_accuracy: 0.9309
Epoch 5/5 loss: 0.0961 - accuracy: 0.9700 - val_loss: 0.2049 - val_accuracy: 0.9344
```
#### Examples End:
#### Tests Start: tagger_cle_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16`
```
loss: 2.2428 - accuracy: 0.3493 - val_loss: 1.8235 - val_accuracy: 0.4233
```
- `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn_cell=LSTM --rnn_cell_dim=16 --cle_dim=16 --word_masking=0.1`
```
loss: 2.2494 - accuracy: 0.3465 - val_loss: 1.8439 - val_accuracy: 0.4232
```
#### Tests End:
