### Assignment: mnist_ensemble
#### Date: Deadline: Mar 13, 7:59 a.m.
#### Points: 2 points
#### Examples: mnist_ensemble_examples
#### Tests: mnist_ensemble_tests

Your goal in this assignment is to implement model ensembling.
The [mnist_ensemble.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_ensemble.py)
template trains `args.models` individual models, and your goal is to perform
an ensemble of the first model, first two models, first three models, …, all
models, and evaluate their accuracy on the test set.

#### Examples Start: mnist_ensemble_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_ensemble.py --models=5`
```
Model 1, individual accuracy 97.84, ensemble accuracy 97.84
Model 2, individual accuracy 98.04, ensemble accuracy 98.22
Model 3, individual accuracy 97.90, ensemble accuracy 98.16
Model 4, individual accuracy 97.88, ensemble accuracy 98.12
Model 5, individual accuracy 97.94, ensemble accuracy 98.12
```
- `python3 mnist_ensemble.py --models=5 --hidden_layers=200`
```
Model 1, individual accuracy 97.78, ensemble accuracy 97.78
Model 2, individual accuracy 98.18, ensemble accuracy 98.30
Model 3, individual accuracy 98.02, ensemble accuracy 98.28
Model 4, individual accuracy 98.10, ensemble accuracy 98.40
Model 5, individual accuracy 97.98, ensemble accuracy 98.44
```
#### Examples End:
#### Tests Start: mnist_ensemble_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_ensemble.py --epochs=1 --models=5`
```
Model 1, individual accuracy 96.24, ensemble accuracy 96.24
Model 2, individual accuracy 96.34, ensemble accuracy 96.44
Model 3, individual accuracy 96.24, ensemble accuracy 96.46
Model 4, individual accuracy 96.64, ensemble accuracy 96.60
Model 5, individual accuracy 96.60, ensemble accuracy 96.60
```
- `python3 mnist_ensemble.py --epochs=1 --models=5 --hidden_layers=200`
```
Model 1, individual accuracy 96.74, ensemble accuracy 96.74
Model 2, individual accuracy 96.92, ensemble accuracy 97.06
Model 3, individual accuracy 96.82, ensemble accuracy 97.06
Model 4, individual accuracy 96.86, ensemble accuracy 96.96
Model 5, individual accuracy 96.46, ensemble accuracy 96.86
```
#### Tests End:
