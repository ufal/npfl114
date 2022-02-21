### Assignment: sgd_manual
#### Date: Deadline: Mar 07, 7:59 a.m.
#### Points: 2 points
#### Examples: sgd_manual_examples

The goal in this exercise is to extend your solution to the
[sgd_backpropagation](https://ufal.mff.cuni.cz/courses/npfl114/2122-summer#sgd_backpropagation)
assignment by **manually** computing the gradient.

While in this assignment we compute the gradient manually, we will nearly always
use the automatic differentiation. Therefore, the assignment is more of
a mathematical exercise than a real-world application. Furthermore, we will
compute the derivatives together on the Mar 16 practicals.

Start with the
[sgd_manual.py](https://github.com/ufal/npfl114/tree/master/labs/02/sgd_manual.py)
template, which is based on
[sgd_backpropagation.py](https://github.com/ufal/npfl114/tree/master/labs/02/sgd_backpropagation.py)
one. Be aware that these templates generates each a different output file.

In order to check that you do not use automatic differentiation, ReCodEx checks
that you do not use `tf.GradientTape` in your solution.

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
