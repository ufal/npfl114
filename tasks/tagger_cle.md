### Assignment: tagger_cle
#### Date: Deadline: Apr 11, 7:59 a.m.
#### Points: 3 points
#### Tests: tagger_cle_tests
#### Examples: tagger_cle_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_cle.py](https://github.com/ufal/npfl114/tree/master/labs/07/tagger_cle.py)
template, implement character-level word embedding computation using
a bidirectional character-level GRU.

Once submitted to ReCodEx, you should experiment with the effect of CLEs
compared to a plain `tagger_we`, and the influence of their dimensionality. Note
that `tagger_cle` has by default smaller word embeddings so that the size
of word representation (64 + 32 + 32) is the same as in the `tagger_we` assignment.

#### Tests Start: tagger_cle_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16 --cle_dim=16`
```
loss: 2.2389 - accuracy: 0.3011 - val_loss: 1.7624 - val_accuracy: 0.4600
```
2. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16 --cle_dim=16 --word_masking=0.1`
```
loss: 2.2506 - accuracy: 0.2967 - val_loss: 1.7892 - val_accuracy: 0.4606
```
#### Tests End:
#### Examples Start: tagger_cle_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_cle.py --epochs=3 --max_sentences=5000 --rnn=LSTM --rnn_dim=32 --cle_dim=32`
```
Epoch 1/3 loss: 0.9835 - accuracy: 0.7045 - val_loss: 0.3117 - val_accuracy: 0.9046
Epoch 2/3 loss: 0.1116 - accuracy: 0.9740 - val_loss: 0.1840 - val_accuracy: 0.9358
Epoch 3/3 loss: 0.0369 - accuracy: 0.9906 - val_loss: 0.1672 - val_accuracy: 0.9394
```
- `python3 tagger_cle.py --epochs=3 --max_sentences=5000 --rnn=LSTM --rnn_dim=32 --cle_dim=32 --word_masking=0.1`
```
Epoch 1/3 loss: 1.0664 - accuracy: 0.6762 - val_loss: 0.3462 - val_accuracy: 0.8997
Epoch 2/3 loss: 0.1977 - accuracy: 0.9475 - val_loss: 0.1834 - val_accuracy: 0.9461
Epoch 3/3 loss: 0.1009 - accuracy: 0.9711 - val_loss: 0.1619 - val_accuracy: 0.9504
```
#### Examples End:
