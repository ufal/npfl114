### Assignment: cags_classification
#### Date: Deadline: Mar 27, 7:59 a.m.
#### Points: 4 points+5 bonus

The goal of this assignment is to use pretrained EfficientNetV2-B0 model to
achieve best accuracy in CAGS classification.

The [CAGS dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/demos/cags_train.html) consists
of images of **ca**ts and do**gs** of size $224×224$, each classified in one of
the 34 breeds and each containing a mask indicating the presence of the animal.
To load the dataset, use the [cags_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/05/cags_dataset.py)
module. The dataset is stored in a
[TFRecord file](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
and each element is encoded as a
[tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example),
which is decoded using the `CAGS.parse` method.

To load the EfficientNetV2-B0, use the
[tf.keras.applications.efficientnet_v2.EfficientNetV2B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet_v2/EfficientNetV2B0)
class, which constructs a Keras model, downloading the weights automatically.

An example performing classification of given images is available in
[image_classification.py](https://github.com/ufal/npfl114/tree/master/labs/05/image_classification.py).

_A note on finetuning: each `tf.keras.layers.Layer` has a mutable `trainable`
property indicating whether its variables should be updated – however, after
changing it, you need to call `.compile` again (or otherwise make sure the list
of trainable variables for the optimizer is updated). Furthermore, `training`
argument passed to the invocation call decides whether the layer is executed in
training regime (neurons gets dropped in dropout, batch normalization computes
estimates on the batch) or in inference regime. There is one exception though
– if `trainable == False` on a batch normalization layer, it runs in the
inference regime even when `training == True`._

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl114/2223-summer#competitions). Everyone who submits a solution
which achieves at least _93%_ test set accuracy will get 4 points; the rest
5 points will be distributed depending on relative ordering of your solutions.

You may want to start with the
[cags_classification.py](https://github.com/ufal/npfl114/tree/master/labs/05/cags_classification.py)
template which generates the test set annotation in the required format.
