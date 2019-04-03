### Assignment: mnist_multiple
#### Date: Deadline: Apr 14, 23:59
#### Points: 4 points

In this assignment you will implement a model with multiple inputs, multiple
outputs, manual batch preparation, and manual evaluation. Start with the
[mnist_multiple.py](https://github.com/ufal/npfl114/tree/master/labs/05/mnist_multiple.py)
template and:
- The goal is to create a model, which given two input MNIST images predicts if the
  digit on the first one is larger than on the second one.
- The model has three outputs:
  - direct prediction of the required value,
  - label prediction for the first image,
  - label prediction for the second image.
- In addition to direct prediction, you can predict labels for both images
  and compare them -- an _indirect prediction_.
- You need to implement:
  - the model, using multiple inputs, outputs, losses, and metrics;
  - generation of two-image batches using regular MNIST batches,
  - computation of direct and indirect prediction accuracy.
