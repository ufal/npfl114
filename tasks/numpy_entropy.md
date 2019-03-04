### Assignment: numpy_entropy
#### Date: Deadline: Mar 17, 23:59
#### Points: 3 points

The goal of this exercise is to famirialize with Python, NumPy and ReCodEx
submission system. Start with the
[numpy_entropy.py](https://github.com/ufal/npfl114/tree/master/labs/01/numpy_entropy.py).

Load a file `numpy_entropy_data.txt`, whose lines consist of data points of our
dataset, and load `numpy_entropy_model.txt`, which describes a model probability distribution,
with each line being a tab-separated pair of _(data point, probability)_.
Example files are in the [labs/01](https://github.com/ufal/npfl114/tree/master/labs/01).

Then compute the following quantities using NumPy, and print them each on
a separate line rounded on two decimal places (or `inf` for positive infinity,
which happens when an element of data distribution has zero probability
under the model distribution):
- entropy _H(data distribution)_
- cross-entropy _H(data distribution, model distribution)_
- KL-divergence _D<sub>KL</sub>(data distribution, model distribution)_

Use natural logarithms to compute the entropies and the divergence.
