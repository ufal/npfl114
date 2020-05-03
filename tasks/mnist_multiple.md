### Assignment: mnist_multiple
#### Date: Deadline: May 03, 23:59
#### Points: 2 points
#### Examples: mnist_multiple_example

In this assignment you will implement a model with multiple inputs. Start with the
[mnist_multiple.py](https://github.com/ufal/npfl114/tree/master/labs/08/mnist_multiple.py)
template and:
- The goal is to create a model, which given two input MNIST images predicts, if the
  digit on the first one is larger than on the second one.
- The model has three outputs:
  - label prediction for the first image,
  - label prediction for the second image,
  - direct prediction whether the first digit is larger than the second one.
- In addition to direct prediction, you can use the predicted labels for both images
  and compare them – an _indirect prediction_.
- You need to implement:
  - the model, using multiple inputs, outputs, losses, and metrics;
  - construction of two-image batches using regular MNIST batches,
  - computation of direct and indirect prediction accuracy.

#### Examples Start: mnist_multiple_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 mnist_multiple.py --recodex --seed=7 --batch_size=50 --epochs=1 --threads=1`
  ```
  93.94 97.32
  ```
- `python3 mnist_multiple.py --recodex --seed=7 --batch_size=100 --epochs=1 --threads=1`
  ```
  91.86 96.28
  ```
#### Examples End:
