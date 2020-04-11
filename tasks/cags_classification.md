### Assignment: cags_classification
#### Date: Deadline: Apr 12, 23:59
#### Points: 6 points+5 bonus

The goal of this assignment is to use pretrained EfficientNet-B0 model to
achieve best accuracy in CAGS classification.

The [CAGS dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/demos/cags_train.html) consists
of images of **ca**ts and **do**gs of size $224×224$, each classified in one of
the 34 breeds and each containing a mask indicating the presence of the animal.
To load the dataset, use the [cags_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/05/cags_dataset.py)
module. The dataset is stored in a
[TFRecord file](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
and each element is encoded as a
[tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example).
Therefore the dataset is loaded using `tf.data` API and each entry can be
decoded using `.map(CAGS.parse)` call.

To load the EfficientNet-B0, use the the provided
[efficient_net.py](https://github.com/ufal/npfl114/tree/master/labs/05/efficient_net.py)
module. Its method `pretrained_efficientnet_b0(include_top)`:
- downloads the pretrained weights if they are not found;
- it returns a `tf.keras.Model` processing image of shape $(224, 224, 3)$ with
  float values in range $[0, 1]$ and producing a list of results:
  - the first value is the final network output:
    - if `include_top == True`, the network will include the final classification
      layer and produce a distribution on 1000 classes (whose names are in
      [imagenet_classes.py](https://github.com/ufal/npfl114/tree/master/labs/05/imagenet_classes.py));
    - if `include_top == False`, the network will return image features (the result
      of the last global average pooling);
  - the rest of outputs are the intermediate results of the network just before
    a convolution with $\textit{stride} > 1$ is performed (denoted $C_5,
    C_4, C_3, C_2, C_1$ in the Object Detection lecture).

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

This is an _open-data task_, where you submit only the test set labels
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

The task is also a [_competition_](#competitions). Everyone who submits
a solution which achieves at least _90%_ test set accuracy will get 6 points;
the rest 5 points will be distributed depending on relative ordering of your
solutions.

You may want to start with the
[cags_classification.py](https://github.com/ufal/npfl114/tree/master/labs/05/cags_classification.py)
template which generates the test set annotation in the required format.
