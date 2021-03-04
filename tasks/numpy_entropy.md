### Assignment: numpy_entropy
#### Date: Deadline: Mar 15, 23:59
#### Points: 3 points
#### Examples: numpy_entropy_example

The goal of this exercise is to familiarize with Python, NumPy and ReCodEx
submission system. Start with the
[numpy_entropy.py](https://github.com/ufal/npfl114/tree/master/labs/01/numpy_entropy.py).

Load a file `numpy_entropy_data.txt`, whose lines consist of data points of our
dataset, and load `numpy_entropy_model.txt`, which describes a model probability distribution,
with each line being a tab-separated pair of _(data point, probability)_.

Then compute the following quantities using NumPy, and print them each on
a separate line rounded on two decimal places (or `inf` for positive infinity,
which happens when an element of data distribution has zero probability
under the model distribution):
- entropy _H(data distribution)_
- cross-entropy _H(data distribution, model distribution)_
- KL-divergence _D<sub>KL</sub>(data distribution, model distribution)_

Use natural logarithms to compute the entropies and the divergence.

#### Examples Start: numpy_entropy_example
For data distribution file [`numpy_entropy_data.txt`](https://github.com/ufal/npfl114/tree/master/labs/01/numpy_entropy_data.txt)
```
A
BB
A
A
BB
A
CCC
```
and model distribution file [`numpy_entropy_model.txt`](https://github.com/ufal/npfl114/tree/master/labs/01/numpy_entropy_model.txt)
```
A	0.5
BB	0.3
CCC	0.1
D	0.1
```
the output should be
```
Entropy: 0.96 nats
Crossentropy: 1.07 nats
KL divergence: 0.11 nats
```
---
If we remove the `CCC	0.1` line from the model distribution, the output should
change to
```
Entropy: 0.96 nats
Crossentropy: inf nats
KL divergence: inf nats
```
#### Examples End:
