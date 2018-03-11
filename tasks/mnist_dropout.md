This exercise evaluates the effect of dropout. Your goal is to modify the
[mnist_dropout.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_dropout.py)
template and implement the following:
- Allow using dropout with specified dropout rate on the hidden layer.
  The dropout must be _active only during training_ and not during test set
  evaluation.
- Print the final accuracy on the test set to standard output. Write the
  accuracy as percentage rounded on two decimal places, e.g., `91.23`.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard (notably training, development
and test set accuracy and loss):
- dropout rate `0`, `0.3`, `0.5`, `0.6`, `0.8`, `0.9`
