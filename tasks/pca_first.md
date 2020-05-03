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
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 pca_first.py --examples=1024 --iterations=64 --seed=7 --threads=1`
  ```
  51.52 9.94
  ```
- `python3 pca_first.py --examples=8192 --iterations=128 --seed=7 --threads=1`
  ```
  52.58 10.20
  ```
- `python3 pca_first.py --examples=55000 --iterations=1024 --seed=7 --threads=1`
  ```
  52.74 9.71
  ```
#### Examples End:
