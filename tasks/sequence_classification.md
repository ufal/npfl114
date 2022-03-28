### Assignment: sequence_classification
#### Date: Deadline: Apr 11, 7:59 a.m.
#### Points: 2 points
#### Examples: sequence_classification_examples
#### Tests: sequence_classification_tests

The goal of this assignment is to introduce recurrent neural networks.
Considering recurrent neural network, the assignment shows convergence speed and
illustrates exploding gradient issue. The network should process sequences of 50
small integers and compute parity for each prefix of the sequence. The inputs
are either 0/1, or vectors with one-hot representation of small integer.

Your goal is to modify the
[sequence_classification.py](https://github.com/ufal/npfl114/tree/master/labs/07/sequence_classification.py)
template and implement the following:
- Use specified RNN type (`SimpleRNN`, `GRU` and `LSTM`) and dimensionality.
- Process the sequence using the required RNN.
- Use additional hidden layer on the RNN outputs if requested.
- Implement gradient clipping if requested.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard
(or [online here](https://tensorboard.dev/experiment/srEoKzfJRBurBrSVNEnLRA/)).
Concentrate on the way how the RNNs converge, convergence speed, exploding
gradient issues and how gradient clipping helps:
- `--rnn_cell=SimpleRNN --sequence_dim=1`, `--rnn_cell=GRU --sequence_dim=1`, `--rnn_cell=LSTM --sequence_dim=1`
- the same as above but with `--sequence_dim=2`
- the same as above but with `--sequence_dim=10`
- `--rnn_cell=SimpleRNN --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30` and the same with `--clip_gradient=1`
- the same as above but with `--rnn_cell=GRU --hidden_layer=75` with and without `--clip_gradient=0.1`
- the same as above but with `--rnn_cell=LSTM --hidden_layer=85` with and without `--clip_gradient=1`

#### Examples Start: sequence_classification_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 sequence_classification.py --rnn_cell=SimpleRNN --epochs=5`
```
Epoch 1/5 loss: 0.6951 - accuracy: 0.5095 - val_loss: 0.6926 - val_accuracy: 0.5176
Epoch 2/5 loss: 0.6924 - accuracy: 0.5158 - val_loss: 0.6921 - val_accuracy: 0.5217
Epoch 3/5 loss: 0.6918 - accuracy: 0.5165 - val_loss: 0.6913 - val_accuracy: 0.5114
Epoch 4/5 loss: 0.6901 - accuracy: 0.5196 - val_loss: 0.6881 - val_accuracy: 0.5157
Epoch 5/5 loss: 0.6842 - accuracy: 0.5220 - val_loss: 0.6793 - val_accuracy: 0.5231
```
- `python3 sequence_classification.py --rnn_cell=GRU --epochs=5`
```
Epoch 1/5 loss: 0.6926 - accuracy: 0.5126 - val_loss: 0.6917 - val_accuracy: 0.5157
Epoch 2/5 loss: 0.6885 - accuracy: 0.5170 - val_loss: 0.6823 - val_accuracy: 0.5143
Epoch 3/5 loss: 0.4987 - accuracy: 0.7328 - val_loss: 0.1574 - val_accuracy: 0.9795
Epoch 4/5 loss: 0.0684 - accuracy: 0.9935 - val_loss: 0.0305 - val_accuracy: 0.9975
Epoch 5/5 loss: 0.0219 - accuracy: 0.9991 - val_loss: 0.0121 - val_accuracy: 0.9998
```
- `python3 sequence_classification.py --rnn_cell=LSTM --epochs=5`
```
Epoch 1/5 loss: 0.6929 - accuracy: 0.5130 - val_loss: 0.6927 - val_accuracy: 0.5153
Epoch 2/5 loss: 0.6919 - accuracy: 0.5155 - val_loss: 0.6902 - val_accuracy: 0.5156
Epoch 3/5 loss: 0.6837 - accuracy: 0.5192 - val_loss: 0.6748 - val_accuracy: 0.5285
Epoch 4/5 loss: 0.3839 - accuracy: 0.7918 - val_loss: 0.0695 - val_accuracy: 1.0000
Epoch 5/5 loss: 0.0351 - accuracy: 1.0000 - val_loss: 0.0183 - val_accuracy: 1.0000
```
- `python3 sequence_classification.py --rnn_cell=LSTM --epochs=5 --hidden_layer=50`
```
Epoch 1/5 loss: 0.6807 - accuracy: 0.5193 - val_loss: 0.6615 - val_accuracy: 0.5233
Epoch 2/5 loss: 0.6485 - accuracy: 0.5373 - val_loss: 0.6378 - val_accuracy: 0.5309
Epoch 3/5 loss: 0.6204 - accuracy: 0.5641 - val_loss: 0.5772 - val_accuracy: 0.6306
Epoch 4/5 loss: 0.0874 - accuracy: 0.9566 - val_loss: 0.0015 - val_accuracy: 1.0000
Epoch 5/5 loss: 8.0165e-04 - accuracy: 1.0000 - val_loss: 3.8375e-04 - val_accuracy: 1.0000
```
- `python3 sequence_classification.py --rnn_cell=LSTM --epochs=5 --hidden_layer=50 --clip_gradient=0.01`
```
Epoch 1/5 loss: 0.6818 - accuracy: 0.5173 - val_loss: 0.6676 - val_accuracy: 0.5241
Epoch 2/5 loss: 0.6509 - accuracy: 0.5374 - val_loss: 0.6393 - val_accuracy: 0.5448
Epoch 3/5 loss: 0.6301 - accuracy: 0.5458 - val_loss: 0.6148 - val_accuracy: 0.5622
Epoch 4/5 loss: 0.5852 - accuracy: 0.6121 - val_loss: 0.4589 - val_accuracy: 0.7884
Epoch 5/5 loss: 0.0372 - accuracy: 0.9881 - val_loss: 0.0060 - val_accuracy: 0.9993
```
#### Examples End:
#### Tests Start: sequence_classification_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn_cell=SimpleRNN --epochs=5`
```
Epoch 1/5 loss: 0.7125 - accuracy: 0.4996 - val_loss: 0.6997 - val_accuracy: 0.4929
Epoch 2/5 loss: 0.6962 - accuracy: 0.4948 - val_loss: 0.6935 - val_accuracy: 0.4985
Epoch 3/5 loss: 0.6931 - accuracy: 0.5155 - val_loss: 0.6922 - val_accuracy: 0.5264
Epoch 4/5 loss: 0.6923 - accuracy: 0.5286 - val_loss: 0.6917 - val_accuracy: 0.5362
Epoch 5/5 loss: 0.6917 - accuracy: 0.5343 - val_loss: 0.6913 - val_accuracy: 0.5323
```
- `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn_cell=GRU --epochs=5`
```
Epoch 1/5 loss: 0.6926 - accuracy: 0.5243 - val_loss: 0.6922 - val_accuracy: 0.5217
Epoch 2/5 loss: 0.6922 - accuracy: 0.5210 - val_loss: 0.6920 - val_accuracy: 0.5217
Epoch 3/5 loss: 0.6919 - accuracy: 0.5247 - val_loss: 0.6916 - val_accuracy: 0.5217
Epoch 4/5 loss: 0.6917 - accuracy: 0.5301 - val_loss: 0.6913 - val_accuracy: 0.5217
Epoch 5/5 loss: 0.6912 - accuracy: 0.5276 - val_loss: 0.6908 - val_accuracy: 0.5220
```
- `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn_cell=LSTM --epochs=5`
```
Epoch 1/5 loss: 0.6928 - accuracy: 0.5358 - val_loss: 0.6925 - val_accuracy: 0.5339
Epoch 2/5 loss: 0.6926 - accuracy: 0.5319 - val_loss: 0.6924 - val_accuracy: 0.5279
Epoch 3/5 loss: 0.6925 - accuracy: 0.5298 - val_loss: 0.6923 - val_accuracy: 0.5343
Epoch 4/5 loss: 0.6924 - accuracy: 0.5332 - val_loss: 0.6922 - val_accuracy: 0.5297
Epoch 5/5 loss: 0.6922 - accuracy: 0.5358 - val_loss: 0.6920 - val_accuracy: 0.5293
```
- `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn_cell=LSTM --epochs=5 --hidden_layer=50`
```
Epoch 1/5 loss: 0.6917 - accuracy: 0.5434 - val_loss: 0.6903 - val_accuracy: 0.5306
Epoch 2/5 loss: 0.6876 - accuracy: 0.5395 - val_loss: 0.6843 - val_accuracy: 0.5350
Epoch 3/5 loss: 0.6784 - accuracy: 0.5550 - val_loss: 0.6732 - val_accuracy: 0.5350
Epoch 4/5 loss: 0.6667 - accuracy: 0.5549 - val_loss: 0.6620 - val_accuracy: 0.5299
Epoch 5/5 loss: 0.6547 - accuracy: 0.5597 - val_loss: 0.6508 - val_accuracy: 0.5278
```
- `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn_cell=LSTM --epochs=5 --hidden_layer=50 --clip_gradient=0.01`
```
Epoch 1/5 loss: 0.6916 - accuracy: 0.5417 - val_loss: 0.6903 - val_accuracy: 0.5308
Epoch 2/5 loss: 0.6876 - accuracy: 0.5390 - val_loss: 0.6844 - val_accuracy: 0.5305
Epoch 3/5 loss: 0.6789 - accuracy: 0.5533 - val_loss: 0.6742 - val_accuracy: 0.5333
Epoch 4/5 loss: 0.6675 - accuracy: 0.5512 - val_loss: 0.6629 - val_accuracy: 0.5411
Epoch 5/5 loss: 0.6563 - accuracy: 0.5532 - val_loss: 0.6536 - val_accuracy: 0.5332
```
#### Tests End:
