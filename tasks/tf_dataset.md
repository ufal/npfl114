### Assignment: tf_dataset
#### Date: Deadline: Apr 05, 23:59
#### Points: 2 points
#### Examples: tf_dataset_examples

In this assignment you will familiarize yourselves with `tf.data`, which is
TensorFlow high-level API for constructing input pipelines. If you want,
you can read an [official TensorFlow tf.data guide](https://www.tensorflow.org/guide/data)
or [reference API manual](https://www.tensorflow.org/api_docs/python/tf/data).

The goal of this assignment is to implement image augmentation preprocessing
similar to `image_augmentation`, but with `tf.data`. Start with the
[tf_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/04/tf_dataset.py)
template and implement the input pipelines employing the `tf.data.Dataset`.
#### Examples Start: tf_dataset_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 tf_dataset.py --batch_size=50`
```
Epoch 1/5 loss: 2.2395 - accuracy: 0.1408 - val_loss: 1.9160 - val_accuracy: 0.3000
Epoch 2/5 loss: 1.9410 - accuracy: 0.2794 - val_loss: 1.7881 - val_accuracy: 0.3430
Epoch 3/5 loss: 1.8415 - accuracy: 0.3287 - val_loss: 1.6749 - val_accuracy: 0.3740
Epoch 4/5 loss: 1.7689 - accuracy: 0.3480 - val_loss: 1.6263 - val_accuracy: 0.3780
Epoch 5/5 loss: 1.7185 - accuracy: 0.3634 - val_loss: 1.5976 - val_accuracy: 0.4260
```
- `python3 tf_dataset.py --batch_size=100`
```
Epoch 1/5 loss: 2.2697 - accuracy: 0.1305 - val_loss: 2.0089 - val_accuracy: 0.2700
Epoch 2/5 loss: 2.0114 - accuracy: 0.2545 - val_loss: 1.8020 - val_accuracy: 0.3410
Epoch 3/5 loss: 1.8473 - accuracy: 0.3278 - val_loss: 1.7071 - val_accuracy: 0.3630
Epoch 4/5 loss: 1.7961 - accuracy: 0.3472 - val_loss: 1.6509 - val_accuracy: 0.3840
Epoch 5/5 loss: 1.7164 - accuracy: 0.3681 - val_loss: 1.6429 - val_accuracy: 0.3910
```
#### Examples End:
