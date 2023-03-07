### Assignment: tf_dataset
#### Date: Deadline: Mar 20, 7:59 a.m.
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
- `python3 tf_dataset.py --epochs=1 --batch_size=100`
```
loss: 2.1809 - accuracy: 0.1772 - val_loss: 1.9773 - val_accuracy: 0.2630
```
- `python3 tf_dataset.py --epochs=1 --batch_size=50 --augment=tf_image`
```
loss: 2.1008 - accuracy: 0.2052 - val_loss: 1.8225 - val_accuracy: 0.3070
```
- `python3 tf_dataset.py --epochs=1 --batch_size=50 --augment=layers`
```
loss: 2.1820 - accuracy: 0.1664 - val_loss: 2.0104 - val_accuracy: 0.2330
```
#### Tests End:
