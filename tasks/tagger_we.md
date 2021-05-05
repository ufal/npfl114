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
- `python3 tagger_we.py --max_sentences=5000 --rnn_cell=LSTM --rnn_cell_dim=16`
```
Epoch 1/5 loss: 1.9780 - accuracy: 0.4436 - val_loss: 0.5346 - val_accuracy: 0.8354
Epoch 2/5 loss: 0.2443 - accuracy: 0.9513 - val_loss: 0.3686 - val_accuracy: 0.8563
Epoch 3/5 loss: 0.0557 - accuracy: 0.9893 - val_loss: 0.3289 - val_accuracy: 0.8735
Epoch 4/5 loss: 0.0333 - accuracy: 0.9916 - val_loss: 0.3430 - val_accuracy: 0.8671
Epoch 5/5 loss: 0.0258 - accuracy: 0.9936 - val_loss: 0.3343 - val_accuracy: 0.8736
loss: 0.3486 - accuracy: 0.8737
```
- `python3 tagger_we.py --max_sentences=5000 --rnn_cell=GRU --rnn_cell_dim=16`
```
Epoch 1/5 loss: 1.6714 - accuracy: 0.5524 - val_loss: 0.3901 - val_accuracy: 0.8744
Epoch 2/5 loss: 0.1312 - accuracy: 0.9722 - val_loss: 0.3210 - val_accuracy: 0.8710
Epoch 3/5 loss: 0.0385 - accuracy: 0.9898 - val_loss: 0.3104 - val_accuracy: 0.8817
Epoch 4/5 loss: 0.0261 - accuracy: 0.9920 - val_loss: 0.3056 - val_accuracy: 0.8886
Epoch 5/5 loss: 0.0210 - accuracy: 0.9933 - val_loss: 0.3052 - val_accuracy: 0.8925
loss: 0.3525 - accuracy: 0.8788
```
#### Examples End:
