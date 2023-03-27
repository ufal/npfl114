### Assignment: tagger_we
#### Date: Deadline: Apr 11, 7:59 a.m.
#### Points: 3 points
#### Tests: tagger_we_tests
#### Examples: tagger_we_examples

In this assignment you will create a simple part-of-speech tagger. For training
and evaluation, we will use Czech dataset containing tokenized sentences, each
word annotated by gold lemma and part-of-speech tag. The
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/07/morpho_dataset.py)
module (down)loads the dataset and provides mappings between strings and integers.

Your goal is to modify the
[tagger_we.py](https://github.com/ufal/npfl114/tree/master/labs/07/tagger_we.py)
template and implement the following:
- Use specified RNN layer type (`GRU` and `LSTM`) and dimensionality.
- Create word embeddings for training vocabulary.
- Process the sentences using bidirectional RNN.
- Predict part-of-speech tags.
Note that you need to properly handle sentences of different lengths in one
batch using [tf.RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor)s.

#### Tests Start: tagger_we_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 tagger_we.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16`
```
loss: 2.3505 - accuracy: 0.2911 - val_loss: 1.9399 - val_accuracy: 0.4305
```
2. `python3 tagger_we.py --epochs=1 --max_sentences=1000 --rnn=GRU --rnn_dim=16`
```
loss: 2.1355 - accuracy: 0.4300 - val_loss: 1.4387 - val_accuracy: 0.5663
```
#### Tests End:
#### Examples Start: tagger_we_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_we.py --epochs=3 --max_sentences=5000 --rnn=LSTM --rnn_dim=64`
```
Epoch 1/3 loss: 0.9607 - accuracy: 0.7078 - val_loss: 0.3827 - val_accuracy: 0.8756
Epoch 2/3 loss: 0.1007 - accuracy: 0.9725 - val_loss: 0.2948 - val_accuracy: 0.8972
Epoch 3/3 loss: 0.0256 - accuracy: 0.9931 - val_loss: 0.2844 - val_accuracy: 0.9024
```
- `python3 tagger_we.py --epochs=3 --max_sentences=5000 --rnn=GRU --rnn_dim=64`
```
Epoch 1/3 loss: 0.7540 - accuracy: 0.7717 - val_loss: 0.3682 - val_accuracy: 0.8712
Epoch 2/3 loss: 0.0726 - accuracy: 0.9797 - val_loss: 0.3989 - val_accuracy: 0.8639
Epoch 3/3 loss: 0.0236 - accuracy: 0.9926 - val_loss: 0.3725 - val_accuracy: 0.8772
```
#### Examples End:
