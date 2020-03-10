### Assignment: mnist_ensemble
#### Date: Deadline: Mar 22, 23:59
#### Points: 2 points
#### Examples: mnist_ensemble_example

Your goal in this assignment is to implement model ensembling.
The [mnist_ensemble.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_ensemble.py)
template trains `args.models` individual models, and your goal is to perform
an ensemble of the first model, first two models, first three models, …, all
models, and evaluate their accuracy on the **development set**.

#### Examples Start: mnist_ensemble_example
_Note that the results might be slightly different, depending on whether you use
GPU or on your CPU type._

Running
```
python3 mnist_ensemble.py --batch_size=50 --epochs=10 --hidden_layers=200 --models=7 --threads=1
```
should give you
```
98.22 98.22
98.04 98.28
98.06 98.34
98.02 98.48
98.24 98.50
98.24 98.60
97.56 98.44
```

Note that how the averaged performance increases with the number of models. Also
note that the seventh model with bad individual accuracy hurts the ensemble.
#### Examples End:
