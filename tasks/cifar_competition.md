### Assignment: cifar_competition
#### Date: Deadline: Apr 05, 23:59
#### Points: 5 points+5 bonus

The goal of this assignment is to devise the best possible model for CIFAR-10.
You can load the data using the
[cifar10.py](https://github.com/ufal/npfl114/tree/master/labs/04/cifar10.py)
module. Note that the test set is different than that of official CIFAR-10.

This is an _open-data task_, where you submit only the test set labels
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

The task is also a [_competition_](#competitions). Everyone who submits
a solution which achieves at least _60%_ test set accuracy will get 5 points;
the rest 5 points will be distributed depending on relative ordering of your
solutions. Note that my solutions usually need to achieve at least ~73% on the
development set to score 60% on the test set.

You may want to start with the
[cifar_competition.py](https://github.com/ufal/npfl114/tree/master/labs/04/cifar_competition.py)
template which generates the test set annotation in the required format.
