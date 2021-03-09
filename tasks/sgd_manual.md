### Assignment: sgd_manual
#### Date: Deadline: Mar 22, 23:59
#### Points: 2 points
#### Examples: sgd_manual_examples

The goal in this exercise is to extend your solution to the
[sgd_backpropagation](#sgd_backpropagation) assignment by **manually**
computing the gradient.

Note that this assignment is the only one where we will compute the gradient
manually, we will otherwise always use the automatic differentiation. Therefore,
the assignment is more of a mathematical exercise and it is definitely not
required to pass the course. Furthermore, we will compute the derivative of the
output functions later on the Mar 9 lecture.

Start with the
[sgd_manual.py](https://github.com/ufal/npfl114/tree/master/labs/02/sgd_manual.py)
template, which is based on
[sgd_backpropagation.py](https://github.com/ufal/npfl114/tree/master/labs/02/sgd_backpropagation.py)
one. Be aware that these templates generates each a different output file.

In order to check that you do not use automatic differentiation, ReCodEx checks
that there is no `GradientTape` string in your source (except in the comments).

#### Examples Start: sgd_manual_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 sgd_manual.py --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.84
Dev accuracy after epoch 2 is 93.86
Dev accuracy after epoch 3 is 94.64
Dev accuracy after epoch 4 is 95.24
Dev accuracy after epoch 5 is 95.26
Test accuracy after epoch 5 is 94.60
```
- `python3 sgd_manual.py --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.66
Dev accuracy after epoch 2 is 95.00
Dev accuracy after epoch 3 is 95.72
Dev accuracy after epoch 4 is 95.80
Dev accuracy after epoch 5 is 96.34
Test accuracy after epoch 5 is 95.31
```
#### Examples End:
