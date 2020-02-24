### Assignment: pca_first
#### Date: Deadline: Mar 8, 23:59
#### Points: 2 points
#### Examples: pca_first_example

The goal of this exercise is to familiarize with TensorFlow `tf.Tensor`s,
shapes and basic tensor manipulation methods. Start with the
[pca_first.py](https://github.com/ufal/npfl114/tree/master/labs/01/pca_first.py).

In this assignment, you will compute the covariance matrix of several examples
from the MNIST dataset, compute the first principal component and quantify
the explained variance of it.

It is fine if you are not familiar with terms like covariance matrix or
principal component – the template contains a detailed description of what
you have to do.

#### Examples Start: pca_first_example
For command
```
python3 pca_first.py --examples=5 --iterations=10
```
the output file `pca_first.out` should contain
```
42.44 34.43
```
---
For command
```
python3 pca_first.py --examples=1000 --iterations=100
```
the output file `pca_first.out` should contain
```
51.59 9.93
```
#### Examples End:
