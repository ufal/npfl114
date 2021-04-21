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
Epoch 1/5 loss: 2.2169 - accuracy: 0.3791 - val_loss: 1.0367 - val_accuracy: 0.7134
Epoch 2/5 loss: 0.7663 - accuracy: 0.8024 - val_loss: 0.5674 - val_accuracy: 0.8317
Epoch 3/5 loss: 0.2492 - accuracy: 0.9591 - val_loss: 0.4647 - val_accuracy: 0.8536
Epoch 4/5 loss: 0.1090 - accuracy: 0.9877 - val_loss: 0.4350 - val_accuracy: 0.8522
Epoch 5/5 loss: 0.0613 - accuracy: 0.9929 - val_loss: 0.4249 - val_accuracy: 0.8549
loss: 0.4313 - accuracy: 0.8561
```
- `python3 tagger_we.py --rnn_cell=GRU --rnn_cell_dim=16`
```
Epoch 1/5 loss: 1.9589 - accuracy: 0.4997 - val_loss: 0.5859 - val_accuracy: 0.8221
Epoch 2/5 loss: 0.3239 - accuracy: 0.9230 - val_loss: 0.3844 - val_accuracy: 0.8455
Epoch 3/5 loss: 0.0672 - accuracy: 0.9884 - val_loss: 0.3782 - val_accuracy: 0.8485
Epoch 4/5 loss: 0.0383 - accuracy: 0.9922 - val_loss: 0.3822 - val_accuracy: 0.8517
Epoch 5/5 loss: 0.0281 - accuracy: 0.9934 - val_loss: 0.3891 - val_accuracy: 0.8570
loss: 0.4329 - accuracy: 0.8457
```
#### Examples End:
