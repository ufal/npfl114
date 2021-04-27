### Assignment: tagger_we
#### Date: Deadline: May 03, 23:59
#### Points: 3 points
#### Examples: tagger_we_examples

In this assignment you will create a simple part-of-speech tagger. For training
and evaluation, we will use Czech dataset containing tokenized sentences, each
word annotated by gold lemma and part-of-speech tag. The
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_dataset.py)
module (down)loads the dataset and provides mappings between strings and integers.

Your goal is to modify the
[tagger_we.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_we.py)
template and implement the following:
- Use specified RNN cell type (`GRU` and `LSTM`) and dimensionality.
- Create word embeddings for training vocabulary.
- Process the sentences using bidirectional RNN.
- Predict part-of-speech tags.
Note that you need to properly handle sentences of different lengths in one
batch using [tf.RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor)s.

#### Examples Start: tagger_we_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 tagger_we.py --rnn_cell=LSTM --rnn_cell_dim=16`
```
Epoch 1/5 loss: 2.1814 - accuracy: 0.3864 - val_loss: 0.9151 - val_accuracy: 0.7349
Epoch 2/5 loss: 0.5976 - accuracy: 0.8584 - val_loss: 0.4956 - val_accuracy: 0.8476
Epoch 3/5 loss: 0.1757 - accuracy: 0.9760 - val_loss: 0.4173 - val_accuracy: 0.8599
Epoch 4/5 loss: 0.0828 - accuracy: 0.9878 - val_loss: 0.4051 - val_accuracy: 0.8612
Epoch 5/5 loss: 0.0556 - accuracy: 0.9902 - val_loss: 0.4099 - val_accuracy: 0.8624
loss: 0.4245 - accuracy: 0.8609
```
- `python3 tagger_we.py --rnn_cell=GRU --rnn_cell_dim=16`
```
Epoch 1/5 loss: 1.8776 - accuracy: 0.5072 - val_loss: 0.5367 - val_accuracy: 0.8179
Epoch 2/5 loss: 0.2400 - accuracy: 0.9546 - val_loss: 0.3917 - val_accuracy: 0.8402
Epoch 3/5 loss: 0.0646 - accuracy: 0.9883 - val_loss: 0.4043 - val_accuracy: 0.8424
Epoch 4/5 loss: 0.0409 - accuracy: 0.9906 - val_loss: 0.4060 - val_accuracy: 0.8438
Epoch 5/5 loss: 0.0343 - accuracy: 0.9910 - val_loss: 0.4015 - val_accuracy: 0.8499
loss: 0.4541 - accuracy: 0.8325
```
#### Examples End:
