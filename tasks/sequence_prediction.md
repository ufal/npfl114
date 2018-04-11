The motivation of this exercise is to learn low-level handling of RNN cells. The network
should learn to predict one specific sequence of
[montly totals of international airline passengers from 1949-1960](https://github.com/ufal/npfl114/tree/master/labs/07/international-airline-passengers.tsv).

Your goal is to modify the
[sequence_prediction.py](https://github.com/ufal/npfl114/tree/master/labs/07/sequence_prediction.py)
template and implement the following:
- Use specified RNN cell type (`RNN`, `GRU` and `LSTM`) and dimensionality.
- For the training part of the sequence, the network should sequentially
  predict the elements, using the correct previous element value as inputs.
- For the testing part of the sequence, the network should sequentially predict
  the elements using its own previous prediction.
- After each epoch, print the `tf.losses.mean_squared_error` of the test part
  prediction using the `"{:.2g}"` format.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard. Note that the network does
not regularize and only uses one sequence, so it is quite brittle.
- try `RNN`, `GRU` and `LSTM` cells
- try dimensions of 5, 10 and 50
