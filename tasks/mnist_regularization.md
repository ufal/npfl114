### Assignment: mnist_regularization
#### Date: Deadline: Mar 22, 23:59
#### Points: 6 points
#### Examples: mnist_regularization_example

You will learn how to implement three regularization methods in this assignment.
Start with the
[mnist_regularization.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_regularization.py)
template and implement the following:
- Allow using dropout with rate `args.dropout`. Add a dropout layer after the
  first `Flatten` and also after all `Dense` hidden layers (but not after the
  output layer).
- Allow using L2 regularization with weight `args.l2`. Use
  `tf.keras.regularizers.L1L2` as a regularizer for all kernels (but not
  biases) of all `Dense` layers (including the last one).
- Allow using label smoothing with weight `args.label_smoothing`. Instead
  of `SparseCategoricalCrossentropy`, you will need to use
  `CategoricalCrossentropy` which offers `label_smoothing` argument.

In ReCodEx, there will be three tests (one for each regularization methods) and
you will get 2 points for passing each one.

In addition to submitting the task in ReCodEx, also run the following
variations and observe the results in TensorBoard (notably training, development
and test set accuracy and loss):
- dropout rate `0`, `0.3`, `0.5`, `0.6`, `0.8`;
- l2 regularization `0`, `0.001`, `0.0001`, `0.00001`;
- label smoothing `0`, `0.1`, `0.3`, `0.5`.

#### Examples Start: mnist_regularization_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 mnist_regularization.py --recodex --seed=7 --threads=1 --epochs=10 --batch_size=50 --hidden_layers=20 --dropout 0.2`
  ```
  90.00
  ```
- `python3 mnist_regularization.py --recodex --seed=7 --threads=1 --epochs=10 --batch_size=50 --hidden_layers=20 --l2 0.01`
  ```
  89.05
  ```
- `python3 mnist_regularization.py --recodex --seed=7 --threads=1 --epochs=10 --batch_size=50 --hidden_layers=20 --label_smoothing 0.2`
  ```
  91.09
  ```
#### Examples End:
