### Assignment: sgd_manual
#### Date: Deadline: Mar ~~15~~ 22, 23:59
#### Points: 2 points
#### Examples: sgd_manual_example

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

#### Examples Start: sgd_manual_example
The outputs should be exactly the same as in the
[sgd_backpropagation](#sgd_backpropagation) assignment.
#### Examples End:
