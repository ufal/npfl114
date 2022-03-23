### Assignment: cnn_manual
#### Date: Deadline: Mar 28, 7:59
#### Points: 3 points
#### Slides: https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/slides/?cnn_manual
#### Video: https://lectures.ms.mff.cuni.cz/video/rec/npfl114/2122/npfl114-cnn_manual.mp4
#### Tests: cnn_manual_tests

To pass this assignment, you need to manually implement the forward and backward
pass through a 2D convolutional layer. Start with the
[cnn_manual.py](https://github.com/ufal/npfl114/tree/master/labs/05/cnn_manual.py)
template, which construct a series of 2D convolutional layers with ReLU
activation and `valid` padding, specified in the `args.cnn` option.
The `args.cnn` contains comma separater layer specifications in the format
`filters-kernel_size-stride`.

Of course, you cannot use any TensorFlow convolutional operation (instead,
implement the forward and backward pass using matrix multiplication and other
operations), nor the `tf.GradientTape` for gradient computation.

To make debugging easier, the template supports a `--verify` option, which
allows comparing the forward pass and the three gradients you compute in the
backward pass to correct values.

#### Tests Start: cnn_manual_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 cnn_manual.py --epochs=1 --cnn=5-1-1`
```
Dev accuracy after epoch 1 is 91.06
Test accuracy after epoch 1 is 89.51
```
- `python3 cnn_manual.py --epochs=1 --cnn=5-3-1`
```
Dev accuracy after epoch 1 is 94.08
Test accuracy after epoch 1 is 92.65
```
- `python3 cnn_manual.py --epochs=1 --cnn=5-3-2`
```
Dev accuracy after epoch 1 is 91.82
Test accuracy after epoch 1 is 90.00
```
- `python3 cnn_manual.py --epochs=1 --cnn=5-3-2,10-3-2`
```
Dev accuracy after epoch 1 is 93.22
Test accuracy after epoch 1 is 91.31
```
#### Tests End:
