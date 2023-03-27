### Assignment: sequence_classification
#### Date: Deadline: Apr 11, 7:59 a.m.
#### Points: 2 points
#### Tests: sequence_classification_tests
#### Examples: sequence_classification_examples

The goal of this assignment is to introduce recurrent neural networks, show
their convergence speed, and illustrate exploding gradient issue. The network
should process sequences of 50 small integers and compute parity for each prefix
of the sequence. The inputs are either 0/1, or vectors with one-hot
representation of small integer.

Your goal is to modify the
[sequence_classification.py](https://github.com/ufal/npfl114/tree/master/labs/07/sequence_classification.py)
template and implement the following:
- Use the specified RNN type (`SimpleRNN`, `GRU`, and `LSTM`) and dimensionality.
- Process the sequence using the required RNN.
- Use additional hidden layer on the RNN outputs if requested.
- Implement gradient clipping if requested.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard
(or [online here](https://tensorboard.dev/experiment/6Zu3udO0R3m5KQDsrqE8jw/)).
Concentrate on the way how the RNNs converge, convergence speed, exploding
gradient issues and how gradient clipping helps:
- `--rnn=SimpleRNN --sequence_dim=1`, `--rnn=GRU --sequence_dim=1`, `--rnn=LSTM --sequence_dim=1`
- the same as above but with `--sequence_dim=3`
- the same as above but with `--sequence_dim=10`
- `--rnn=SimpleRNN --hidden_layer=85 --rnn_dim=30 --sequence_dim=30` and the same with `--clip_gradient=1`
- the same as above but with `--rnn=GRU` with and without `--clip_gradient=1`
- the same as above but with `--rnn=LSTM` with and without `--clip_gradient=1`

#### Tests Start: sequence_classification_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=SimpleRNN --epochs=5`
```
Epoch 1/5 loss: 0.6996 - accuracy: 0.4572 - val_loss: 0.6958 - val_accuracy: 0.4485
Epoch 2/5 loss: 0.6931 - accuracy: 0.5275 - val_loss: 0.6930 - val_accuracy: 0.5257
Epoch 3/5 loss: 0.6913 - accuracy: 0.5480 - val_loss: 0.6914 - val_accuracy: 0.5398
Epoch 4/5 loss: 0.6901 - accuracy: 0.5479 - val_loss: 0.6901 - val_accuracy: 0.5523
Epoch 5/5 loss: 0.6887 - accuracy: 0.5493 - val_loss: 0.6886 - val_accuracy: 0.5540
```
2. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=GRU --epochs=5`
```
Epoch 1/5 loss: 0.6942 - accuracy: 0.4766 - val_loss: 0.6934 - val_accuracy: 0.4635
Epoch 2/5 loss: 0.6930 - accuracy: 0.5046 - val_loss: 0.6927 - val_accuracy: 0.5278
Epoch 3/5 loss: 0.6924 - accuracy: 0.5338 - val_loss: 0.6922 - val_accuracy: 0.5331
Epoch 4/5 loss: 0.6921 - accuracy: 0.5307 - val_loss: 0.6918 - val_accuracy: 0.5343
Epoch 5/5 loss: 0.6917 - accuracy: 0.5310 - val_loss: 0.6914 - val_accuracy: 0.5217
```
3. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5`
```
Epoch 1/5 loss: 0.6935 - accuracy: 0.4816 - val_loss: 0.6934 - val_accuracy: 0.4615
Epoch 2/5 loss: 0.6931 - accuracy: 0.4979 - val_loss: 0.6931 - val_accuracy: 0.5250
Epoch 3/5 loss: 0.6929 - accuracy: 0.5264 - val_loss: 0.6929 - val_accuracy: 0.5275
Epoch 4/5 loss: 0.6928 - accuracy: 0.5321 - val_loss: 0.6927 - val_accuracy: 0.5340
Epoch 5/5 loss: 0.6925 - accuracy: 0.5420 - val_loss: 0.6925 - val_accuracy: 0.5357
```
4. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5 --hidden_layer=50`
```
Epoch 1/5 loss: 0.6917 - accuracy: 0.5486 - val_loss: 0.6905 - val_accuracy: 0.5315
Epoch 2/5 loss: 0.6889 - accuracy: 0.5382 - val_loss: 0.6869 - val_accuracy: 0.5274
Epoch 3/5 loss: 0.6832 - accuracy: 0.5451 - val_loss: 0.6792 - val_accuracy: 0.5354
Epoch 4/5 loss: 0.6731 - accuracy: 0.5522 - val_loss: 0.6677 - val_accuracy: 0.5601
Epoch 5/5 loss: 0.6610 - accuracy: 0.5538 - val_loss: 0.6567 - val_accuracy: 0.5613
```
5. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5 --hidden_layer=50 --clip_gradient=0.01`
```
Epoch 1/5 loss: 0.6917 - accuracy: 0.5477 - val_loss: 0.6904 - val_accuracy: 0.5368
Epoch 2/5 loss: 0.6886 - accuracy: 0.5447 - val_loss: 0.6865 - val_accuracy: 0.5347
Epoch 3/5 loss: 0.6828 - accuracy: 0.5476 - val_loss: 0.6786 - val_accuracy: 0.5512
Epoch 4/5 loss: 0.6730 - accuracy: 0.5559 - val_loss: 0.6680 - val_accuracy: 0.5779
Epoch 5/5 loss: 0.6618 - accuracy: 0.5541 - val_loss: 0.6584 - val_accuracy: 0.5344
```
#### Tests End:
#### Examples Start: sequence_classification_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 sequence_classification.py --rnn=SimpleRNN --epochs=5`
```
Epoch 1/5 loss: 0.6925 - accuracy: 0.5138 - val_loss: 0.6890 - val_accuracy: 0.5187
Epoch 2/5 loss: 0.6844 - accuracy: 0.5200 - val_loss: 0.6792 - val_accuracy: 0.5126
Epoch 3/5 loss: 0.6758 - accuracy: 0.5174 - val_loss: 0.6722 - val_accuracy: 0.5169
Epoch 4/5 loss: 0.6705 - accuracy: 0.5146 - val_loss: 0.6681 - val_accuracy: 0.5127
Epoch 5/5 loss: 0.6668 - accuracy: 0.5145 - val_loss: 0.6643 - val_accuracy: 0.5201
```
- `python3 sequence_classification.py --rnn=GRU --epochs=5`
```
Epoch 1/5 loss: 0.6928 - accuracy: 0.5090 - val_loss: 0.6918 - val_accuracy: 0.5160
Epoch 2/5 loss: 0.6868 - accuracy: 0.5160 - val_loss: 0.6762 - val_accuracy: 0.5314
Epoch 3/5 loss: 0.3761 - accuracy: 0.8131 - val_loss: 0.0623 - val_accuracy: 1.0000
Epoch 4/5 loss: 0.0344 - accuracy: 0.9993 - val_loss: 0.0194 - val_accuracy: 0.9996
Epoch 5/5 loss: 0.0137 - accuracy: 0.9997 - val_loss: 0.0085 - val_accuracy: 1.0000
```
- `python3 sequence_classification.py --rnn=LSTM --epochs=5`
```
Epoch 1/5 loss: 0.6931 - accuracy: 0.5063 - val_loss: 0.6929 - val_accuracy: 0.5135
Epoch 2/5 loss: 0.6921 - accuracy: 0.5148 - val_loss: 0.6900 - val_accuracy: 0.5137
Epoch 3/5 loss: 0.5484 - accuracy: 0.6868 - val_loss: 0.1687 - val_accuracy: 0.9983
Epoch 4/5 loss: 0.0766 - accuracy: 0.9998 - val_loss: 0.0338 - val_accuracy: 1.0000
Epoch 5/5 loss: 0.0215 - accuracy: 1.0000 - val_loss: 0.0137 - val_accuracy: 1.0000
```
- `python3 sequence_classification.py --rnn=LSTM --epochs=5 --hidden_layer=50`
```
Epoch 1/5 loss: 0.6829 - accuracy: 0.5160 - val_loss: 0.6601 - val_accuracy: 0.5166
Epoch 2/5 loss: 0.6447 - accuracy: 0.5398 - val_loss: 0.6310 - val_accuracy: 0.5274
Epoch 3/5 loss: 0.6226 - accuracy: 0.5545 - val_loss: 0.6108 - val_accuracy: 0.5522
Epoch 4/5 loss: 0.5895 - accuracy: 0.5859 - val_loss: 0.5529 - val_accuracy: 0.6215
Epoch 5/5 loss: 0.4641 - accuracy: 0.7130 - val_loss: 0.3106 - val_accuracy: 0.8517
```
- `python3 sequence_classification.py --rnn=LSTM --epochs=5 --hidden_layer=50 --clip_gradient=1`
```
Epoch 1/5 loss: 0.6829 - accuracy: 0.5160 - val_loss: 0.6601 - val_accuracy: 0.5166
Epoch 2/5 loss: 0.6447 - accuracy: 0.5398 - val_loss: 0.6310 - val_accuracy: 0.5274
Epoch 3/5 loss: 0.6226 - accuracy: 0.5545 - val_loss: 0.6108 - val_accuracy: 0.5522
Epoch 4/5 loss: 0.5892 - accuracy: 0.5859 - val_loss: 0.5516 - val_accuracy: 0.6238
Epoch 5/5 loss: 0.4073 - accuracy: 0.7596 - val_loss: 0.1919 - val_accuracy: 0.9277
```
#### Examples End:
