### Assignment: mnist_ensemble
#### Date: Deadline: Mar 31, 23:59
#### Points: 2 points

Your goal in this assignment is to implement model ensembling.
The [mnist_ensemble.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_ensemble.py)
template trains `args.models` individual models, and your goal is to perform
an ensemble of the first model, first two models, first three models, …, all
models, and evaluate their accuracy on the development set.

In addition to submitting the task in ReCodEx, run the script with
`args.models=7` and look at the results in `mnist_ensemble.out` file.
