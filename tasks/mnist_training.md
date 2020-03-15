### Assignment: mnist_training
#### Date: Deadline: Mar ~~15~~ 22, 23:59
#### Points: 3 points

This exercise should teach you using different optimizers, learning rates,
and learning rate decays. Your goal is to modify the
[mnist_training.py](https://github.com/ufal/npfl114/tree/master/labs/02/mnist_training.py)
template and implement the following:
- Using specified optimizer (either `SGD` or `Adam`).
- Optionally using momentum for the `SGD` optimizer.
- Using specified learning rate for the optimizer.
- Optionally use a given learning rate schedule. The schedule can be either
  `exponential` or `polynomial` (with degree 1, so inverse time decay).
  Additionally, the final learning rate is given and the decay should gradually
  decrease the learning rate to reach the final learning rate just after the
  training.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard:
- `SGD` optimizer, `learning_rate` 0.01;
- `SGD` optimizer, `learning_rate` 0.01, `momentum` 0.9;
- `SGD` optimizer, `learning_rate` 0.1;
- `Adam` optimizer, `learning_rate` 0.001;
- `Adam` optimizer, `learning_rate` 0.01;
- `Adam` optimizer, `exponential` decay, `learning_rate` 0.01 and `learning_rate_final` 0.001;
- `Adam` optimizer, `polynomial` decay, `learning_rate` 0.01 and `learning_rate_final` 0.0001.
