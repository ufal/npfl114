### Assignment: tagger_crf_manual
#### Date: Deadline: Apr 17, 7:59 a.m.
#### Points: 2 points
#### Tests: tagger_crf_manual_tests
#### Examples: tagger_crf_manual_examples

This assignment is an extension of `tagger_crf`, where we will perform the CRF
loss computation (but not the CRF decoding) manually.

The [tagger_crf_manual.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_crf_manual.py)
template is nearly identical to `tagger_crf`, the only difference is the
`crf_loss` method, where you should manually implement the CRF loss.

#### Tests Start: tagger_crf_manual_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 tagger_crf_manual.py --epochs=2 --max_sentences=1000 --rnn=LSTM --rnn_dim=24`
```
Epoch 1/2 loss: 30.0429 - val_loss: 19.9425 - val_f1: 0.0000e+00
Epoch 2/2 loss: 16.9279 - val_loss: 17.7281 - val_f1: 0.0039
```
2. `python3 tagger_crf_manual.py --epochs=2 --max_sentences=1000 --rnn=GRU --rnn_dim=24`
```
Epoch 1/2 loss: 29.0089 - val_loss: 19.2492 - val_f1: 0.0000e+00
Epoch 2/2 loss: 15.3984 - val_loss: 17.9794 - val_f1: 0.0811
```
#### Tests End:
#### Examples Start: tagger_crf_manual_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_crf_manual.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=64`
```
Epoch 1/5 loss: 17.2315 - val_loss: 13.4043 - val_f1: 0.1145
Epoch 2/5 loss: 8.7492 - val_loss: 10.6630 - val_f1: 0.3155
Epoch 3/5 loss: 4.5754 - val_loss: 9.7337 - val_f1: 0.4022
Epoch 4/5 loss: 2.1429 - val_loss: 10.1171 - val_f1: 0.4463
Epoch 5/5 loss: 1.1066 - val_loss: 10.5541 - val_f1: 0.4553
```
- `python3 tagger_crf_manual.py --epochs=5 --max_sentences=5000 --rnn=GRU --rnn_dim=64`
```
Epoch 1/5 loss: 16.4778 - val_loss: 12.8231 - val_f1: 0.2221
Epoch 2/5 loss: 7.2317 - val_loss: 9.5449 - val_f1: 0.4297
Epoch 3/5 loss: 2.8447 - val_loss: 10.2954 - val_f1: 0.4776
Epoch 4/5 loss: 1.1184 - val_loss: 11.5283 - val_f1: 0.4702
Epoch 5/5 loss: 0.5509 - val_loss: 11.1679 - val_f1: 0.4822
```
#### Examples End:
