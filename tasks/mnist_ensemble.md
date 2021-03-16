### Assignment: mnist_ensemble
#### Date: Deadline: Mar 29, 23:59
#### Points: 2 points
#### Examples: mnist_ensemble_examples

Your goal in this assignment is to implement model ensembling.
The [mnist_ensemble.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_ensemble.py)
template trains `args.models` individual models, and your goal is to perform
an ensemble of the first model, first two models, first three models, …, all
models, and evaluate their accuracy on the **development set**.

#### Examples Start: mnist_ensemble_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 mnist_ensemble.py --models=3`
```
Model 1, individual accuracy 97.78, ensemble accuracy 97.78
Model 2, individual accuracy 97.76, ensemble accuracy 98.02
Model 3, individual accuracy 97.88, ensemble accuracy 98.06
```
- `python3 mnist_ensemble.py --models=5`
```
Model 1, individual accuracy 97.78, ensemble accuracy 97.78
Model 2, individual accuracy 97.76, ensemble accuracy 98.02
Model 3, individual accuracy 97.88, ensemble accuracy 98.06
Model 4, individual accuracy 97.78, ensemble accuracy 98.10
Model 5, individual accuracy 97.78, ensemble accuracy 98.10
```
#### Examples End:
