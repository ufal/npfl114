### Assignment: tagger_crf
#### Date: Deadline: May 10, 23:59
#### Points: 2 points
#### Examples: tagger_crf_examples

This assignment is an extension of `tagger_we` task. Using the
[tagger_crf.py](https://github.com/ufal/npfl114/tree/master/labs/09/tagger_crf.py)
template, implement named entity recognition using CRF loss and CRF decoding
from the `tensorflow_addons` package.

The evaluation is performed using the provided metric computing F1 score of the
span prediction (i.e., a recognized possibly-multiword named entity is true
positive if both the entity type and the span exactly match).

In practice, character-level embeddings (and also pre-trained word embeddings)
would be used to obtain superior results.

#### Examples Start: tagger_crf_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 tagger_crf.py --max_sentences=5000 --rnn_cell=LSTM --rnn_cell_dim=24`
```
Epoch 1/5 loss: 18.5475 - val_f1: 0.0248
Epoch 2/5 loss: 9.8655 - val_f1: 0.2207
Epoch 3/5 loss: 6.0053 - val_f1: 0.3370
Epoch 4/5 loss: 3.1784 - val_f1: 0.4000
Epoch 5/5 loss: 1.6535 - val_f1: 0.4363
```
- `python3 tagger_crf.py --max_sentences=5000 --rnn_cell=GRU --rnn_cell_dim=24`
```
Epoch 1/5 loss: 17.7499 - val_f1: 0.1624
Epoch 2/5 loss: 8.3992 - val_f1: 0.4048
Epoch 3/5 loss: 3.7579 - val_f1: 0.4444
Epoch 4/5 loss: 1.5298 - val_f1: 0.4496
Epoch 5/5 loss: 0.7858 - val_f1: 0.4769
```
#### Examples End:
