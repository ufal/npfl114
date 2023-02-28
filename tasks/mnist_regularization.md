### Assignment: mnist_regularization
#### Date: Deadline: Mar 13, 7:59 a.m.
#### Points: 3 points
#### Tests: mnist_regularization_tests

You will learn how to implement three regularization methods in this assignment.
Start with the
[mnist_regularization.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_regularization.py)
template and implement the following:
- Allow using dropout with rate `args.dropout`. Add a dropout layer after the
  first `Flatten` and also after all `Dense` hidden layers (but not after the
  output layer).
- Allow using AdamW with weight decay with strength of `args.weight_decay`,
  making sure the weight decay is not applied on bias.
- Allow using label smoothing with weight `args.label_smoothing`. Instead
  of `SparseCategoricalCrossentropy`, you will need to use
  `CategoricalCrossentropy` which offers `label_smoothing` argument.

In addition to submitting the task in ReCodEx, also run the following
variations and observe the results in TensorBoard
(or [online here](https://tensorboard.dev/experiment/EK3HQIbuQU2CXyT1Cm5wbg/)),
notably the training, development and test set accuracy and loss:
- dropout rate `0`, `0.3`, `0.5`, `0.6`, `0.8`;
- weight decay `0`, `0.1`, `0.3`, `0.5`, `0.1`;
- label smoothing `0`, `0.1`, `0.3`, `0.5`.

#### Tests Start: mnist_regularization_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_regularization.py --epochs=1 --dropout=0.3`
```
loss: 0.7988 - accuracy: 0.7646 - val_loss: 0.3164 - val_accuracy: 0.9116
```
- `python3 mnist_regularization.py --epochs=1 --dropout=0.5 --hidden_layers 300 300`
```
loss: 1.4830 - accuracy: 0.4910 - val_loss: 0.4659 - val_accuracy: 0.8766
```
- `python3 mnist_regularization.py --epochs=1 --weight_decay=0.1`
```
loss: 0.6040 - accuracy: 0.8386 - val_loss: 0.2718 - val_accuracy: 0.9236
```
- `python3 mnist_regularization.py --epochs=1 --weight_decay=0.3`
```
loss: 0.6062 - accuracy: 0.8384 - val_loss: 0.2744 - val_accuracy: 0.9222
```
- `python3 mnist_regularization.py --epochs=1 --label_smoothing=0.1`
```
loss: 0.9926 - accuracy: 0.8414 - val_loss: 0.7720 - val_accuracy: 0.9222
```
- `python3 mnist_regularization.py --epochs=1 --label_smoothing=0.3`
```
loss: 1.5080 - accuracy: 0.8456 - val_loss: 1.3738 - val_accuracy: 0.9260
```
#### Tests End:
