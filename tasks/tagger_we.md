### Assignment: tagger_we
#### Date: Deadline: Apr 11, 7:59 a.m.
#### Points: 3 points
#### Examples: tagger_we_examples
#### Tests: tagger_we_tests

In this assignment you will create a simple part-of-speech tagger. For training
and evaluation, we will use Czech dataset containing tokenized sentences, each
word annotated by gold lemma and part-of-speech tag. The
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/07/morpho_dataset.py)
module (down)loads the dataset and provides mappings between strings and integers.

Your goal is to modify the
[tagger_we.py](https://github.com/ufal/npfl114/tree/master/labs/07/tagger_we.py)
template and implement the following:
- Use specified RNN cell type (`GRU` and `LSTM`) and dimensionality.
- Create word embeddings for training vocabulary.
- Process the sentences using bidirectional RNN.
- Predict part-of-speech tags.
Note that you need to properly handle sentences of different lengths in one
batch using [tf.RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor)s.

#### Examples Start: tagger_we_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_we.py --max_sentences=5000 --rnn_cell=LSTM --rnn_cell_dim=16`
```
Epoch 1/5 loss: 1.3015 - accuracy: 0.6197 - val_loss: 0.5595 - val_accuracy: 0.8438
Epoch 2/5 loss: 0.2059 - accuracy: 0.9561 - val_loss: 0.3388 - val_accuracy: 0.8954
Epoch 3/5 loss: 0.0510 - accuracy: 0.9888 - val_loss: 0.3189 - val_accuracy: 0.8941
Epoch 4/5 loss: 0.0306 - accuracy: 0.9920 - val_loss: 0.3265 - val_accuracy: 0.8916
Epoch 5/5 loss: 0.0213 - accuracy: 0.9947 - val_loss: 0.3260 - val_accuracy: 0.8926
```
- `python3 tagger_we.py --max_sentences=5000 --rnn_cell=GRU --rnn_cell_dim=16`
```
Epoch 1/5 loss: 0.9769 - accuracy: 0.7228 - val_loss: 0.4172 - val_accuracy: 0.8750
Epoch 2/5 loss: 0.1204 - accuracy: 0.9740 - val_loss: 0.3330 - val_accuracy: 0.8852
Epoch 3/5 loss: 0.0365 - accuracy: 0.9900 - val_loss: 0.3138 - val_accuracy: 0.8903
Epoch 4/5 loss: 0.0261 - accuracy: 0.9919 - val_loss: 0.3234 - val_accuracy: 0.8840
Epoch 5/5 loss: 0.0203 - accuracy: 0.9935 - val_loss: 0.3246 - val_accuracy: 0.8837
```
#### Examples End:
#### Tests Start: tagger_we_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_we.py --epochs=1 --max_sentences=1000 --rnn_cell=LSTM --rnn_cell_dim=16`
```
loss: 2.3174 - accuracy: 0.3603 - val_loss: 1.9011 - val_accuracy: 0.4222
```
- `python3 tagger_we.py --epochs=1 --max_sentences=1000 --rnn_cell=GRU --rnn_cell_dim=16`
```
loss: 2.1435 - accuracy: 0.4186 - val_loss: 1.5338 - val_accuracy: 0.5498
```
#### Tests End:
