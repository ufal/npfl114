### Assignment: sequence_classification
#### Date: Deadline: Apr 26, 23:59
#### Points: 6 points

The goal of this assignment is to introduce recurrent neural networks, manual
TensorBoard log collection, and manual gradient clipping. Considering recurrent
neural network, the assignment shows convergence speed and illustrates exploding
gradient issue. The network should process sequences of 50 small integers and
compute parity for each prefix of the sequence. The inputs are either 0/1, or
vectors with one-hot representation of small integer.

Your goal is to modify the
[sequence_classification.py](https://github.com/ufal/npfl114/tree/master/labs/07/sequence_classification.py)
template and implement the following:
- Use specified RNN type (`SimpleRNN`, `GRU` and `LSTM`) and dimensionality.
- Process the sequence using the required RNN.
- Use additional hidden layer on the RNN outputs if requested.
- Implement gradient clipping if requested.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard. Concentrate on the way
how the RNNs converge, convergence speed, exploding gradient issues
and how gradient clipping helps:
- `--rnn_cell=SimpleRNN --sequence_dim=1`, `--rnn_cell=GRU --sequence_dim=1`, `--rnn_cell=LSTM --sequence_dim=1`
- the same as above but with `--sequence_dim=2`
- the same as above but with `--sequence_dim=10`
- `--rnn_cell=LSTM --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30` and the same with `--clip_gradient=1`
- the same as above but with `--rnn_cell=SimpleRNN`
- the same as above but with `--rnn_cell=GRU --hidden_layer=90`
