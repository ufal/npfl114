### Assignment: tf_dataset
#### Date: Deadline: Mar 21, 7:59
#### Points: 2 points
#### Tests: tf_dataset_tests

In this assignment you will familiarize yourselves with `tf.data`, which is
TensorFlow high-level API for constructing input pipelines. If you want,
you can read an [official TensorFlow tf.data guide](https://www.tensorflow.org/guide/data)
or [reference API manual](https://www.tensorflow.org/api_docs/python/tf/data).

The goal of this assignment is to implement image augmentation preprocessing
similar to `image_augmentation`, but with `tf.data`. Start with the
[tf_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/04/tf_dataset.py)
template and implement the input pipelines employing the `tf.data.Dataset`.

#### Tests Start: tf_dataset_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tf_dataset.py --epochs=1 --batch_size=50`
```
loss: 2.1262 - accuracy: 0.1998 - val_loss: 1.8775 - val_accuracy: 0.3040
```
- `python3 tf_dataset.py --epochs=1 --batch_size=100`
```
loss: 2.2113 - accuracy: 0.1618 - val_loss: 2.0246 - val_accuracy: 0.2640
```
#### Tests End:
