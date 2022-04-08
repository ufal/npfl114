### Assignment: tagger_crf
#### Date: Deadline: Apr 19, 7:59 a.m.
#### Points: 2 points
#### Examples: tagger_crf_examples
#### Tests: tagger_crf_tests

This assignment is an extension of `tagger_we` task. Using the
[tagger_crf.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_crf.py)
template, implement named entity recognition using CRF loss and CRF decoding
from the `tensorflow_addons` package.

The evaluation is performed using the provided metric computing F1 score of the
span prediction (i.e., a recognized possibly-multiword named entity is true
positive if both the entity type and the span exactly match).

In practice, character-level embeddings (and also pre-trained word embeddings)
would be used to obtain superior results.

#### Examples Start: tagger_crf_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_crf.py --max_sentences=5000 --rnn_cell=LSTM --rnn_cell_dim=24`
```
Epoch 1/5 loss: 18.5371 - val_loss: 14.0865 - val_f1: 0.0317
Epoch 2/5 loss: 9.7936 - val_loss: 11.5969 - val_f1: 0.2428
Epoch 3/5 loss: 5.9049 - val_loss: 9.8079 - val_f1: 0.3645
Epoch 4/5 loss: 3.1811 - val_loss: 9.5350 - val_f1: 0.4276
Epoch 5/5 loss: 1.7330 - val_loss: 9.2801 - val_f1: 0.4398
```
- `python3 tagger_crf.py --max_sentences=5000 --rnn_cell=GRU --rnn_cell_dim=24`
```
Epoch 1/5 loss: 17.6696 - val_loss: 13.5141 - val_f1: 0.1700
Epoch 2/5 loss: 8.1954 - val_loss: 10.2339 - val_f1: 0.4070
Epoch 3/5 loss: 3.7555 - val_loss: 9.4217 - val_f1: 0.4528
Epoch 4/5 loss: 1.6607 - val_loss: 10.1525 - val_f1: 0.4546
Epoch 5/5 loss: 0.8472 - val_loss: 10.6141 - val_f1: 0.4744
```
#### Examples End:
#### Tests Start: tagger_crf_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_crf.py --epochs=2 --max_sentences=1000 --rnn_cell=LSTM --rnn_cell_dim=24`
```
Epoch 1/2 loss: 29.9874 - val_loss: 20.2837 - val_f1: 0.0000e+00
Epoch 2/2 loss: 17.2559 - val_loss: 18.0548 - val_f1: 0.0030
```
- `python3 tagger_crf.py --epochs=2 --max_sentences=1000 --rnn_cell=GRU --rnn_cell_dim=24`
```
Epoch 1/2 loss: 29.1122 - val_loss: 19.1089 - val_f1: 0.0000e+00
Epoch 2/2 loss: 15.7085 - val_loss: 17.1493 - val_f1: 0.0172
```
#### Tests End:
