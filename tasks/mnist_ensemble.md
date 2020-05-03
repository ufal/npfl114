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
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 mnist_ensemble.py --recodex --seed=7 --threads=1 --epochs=2 --batch_size=50 --hidden_layers=20 --models=3`
  ```
  94.96 94.96
  95.54 95.58
  94.90 95.54
  ```
- `python3 mnist_ensemble.py --recodex --seed=7 --threads=1 --epochs=1 --batch_size=50 --hidden_layers=20 --models=5`
  ```
  94.08 94.08
  94.36 94.34
  93.94 94.20
  94.02 94.20
  93.94 94.16
  ```
#### Examples End:
