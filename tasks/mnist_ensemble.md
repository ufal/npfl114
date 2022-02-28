### Assignment: mnist_ensemble
#### Date: Deadline: Mar 14, 7:59 a.m.
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
Model 1, individual accuracy 97.69, ensemble accuracy 97.69
Model 2, individual accuracy 97.75, ensemble accuracy 98.03
Model 3, individual accuracy 97.90, ensemble accuracy 98.08
Model 4, individual accuracy 97.52, ensemble accuracy 98.05
Model 5, individual accuracy 97.59, ensemble accuracy 98.14
```
- `python3 mnist_ensemble.py --models=5 --hidden_layers=200`
```
Model 1, individual accuracy 97.86, ensemble accuracy 97.86
Model 2, individual accuracy 98.09, ensemble accuracy 98.27
Model 3, individual accuracy 98.15, ensemble accuracy 98.41
Model 4, individual accuracy 98.13, ensemble accuracy 98.45
Model 5, individual accuracy 97.79, ensemble accuracy 98.39
```
#### Examples End:
#### Tests Start: mnist_ensemble_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_ensemble.py --epochs=1 --models=5`
```
Model 1, individual accuracy 95.17, ensemble accuracy 95.17
Model 2, individual accuracy 94.75, ensemble accuracy 95.10
Model 3, individual accuracy 95.19, ensemble accuracy 95.11
Model 4, individual accuracy 95.11, ensemble accuracy 95.13
Model 5, individual accuracy 95.20, ensemble accuracy 95.24
```
- `python3 mnist_ensemble.py --epochs=1 --models=5 --hidden_layers=200`
```
Model 1, individual accuracy 96.05, ensemble accuracy 96.05
Model 2, individual accuracy 96.11, ensemble accuracy 96.21
Model 3, individual accuracy 95.76, ensemble accuracy 96.16
Model 4, individual accuracy 95.85, ensemble accuracy 96.08
Model 5, individual accuracy 95.94, ensemble accuracy 96.10
```
#### Tests End:
