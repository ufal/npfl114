This exercise should teach you using different optimizers and learning rates
(including exponential decay). Your goal is to modify the
[mnist_training.py](https://github.com/ufal/npfl114/tree/master/labs/02/mnist_training.py)
template and implement the following:
- Using specified optimizer (either `SGD` or `Adam`).
- Optionally using momentum for the `SGD` optimizer.
- Using specified initial learning rate for the optimizer.
- Optionally use given final learning rate. If the final learning rate is
  given, implement exponential learning rate decay (using `tf.train.exponential_decay`).
  Specifically, for the whole first epoch, train using the given initial learning rate.
  Then lower the learning rate between epochs by multiplying it each time by
  the same suitable constant, such that the whole last epoch is trained using
  the specified final learning rate.
- Print the final accuracy on the test set to standard output. Write the
  accuracy as percentage rounded on two decimal places, e.g., `91.23`.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard:
- `SGD` optimizer, `learning_rate` 0.01;
- `SGD` optimizer, `learning_rate` 0.01, `momentum` 0.9;
- `SGD` optimizer, `learning_rate` 0.1;
- `Adam` optimizer, `learning_rate` 0.001;
- `Adam` optimizer, `learning_rate` 0.01 and `learning_rate_final` 0.001.
