### Assignment: svhn_competition
#### Date: Deadline: ~~Apr 26, 23:59~~ May 05, 23:59
#### Points: 5 points+5 bonus

The goal of this assignment is to implement a system performing object
recognition, optionally utilizing pretrained EfficientNet-B0 backbone.

The [Street View House Numbers (SVHN) dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/demos/svhn_train.html)
annotates for every photo all digits appearing on it, including their bounding
boxes. The dataset can be loaded using the [svhn_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/06/svhn_dataset.py)
module. Similarly to the `CAGS` dataset, it is stored in a
[TFRecord file](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
with [tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
elements, which can be decoded using `.map(SVHN.parse)` call. Every element
is a dictionary with the following keys:
- `"image"`: a square 3-channel image,
- `"classes"`: a 1D tensor with all digit labels appearing in the image,
- `"bboxes"`: a `[num_digits, 4]` 2D tensor with bounding boxes of every
  digit in the image.

Given that the dataset elements are each of possibly different size and you
want to preprocess them using a NumPy function `bboxes_training`, it might be
more comfortable to convert the dataset to NumPy. Alternatively, you can
call `bboxes_training` directly in `tf.data.Dataset.map` by using `tf.numpy_function`,
see [FAQ](#faq).

Similarly to the `cags_classification`, you can load the EfficientNet-B0 using the provided
[efficient_net.py](https://github.com/ufal/npfl114/tree/master/labs/06/efficient_net.py)
module. Its method `pretrained_efficientnet_b0(include_top, dynamic_shape=False)` has gotten
a new argument `dynamic_shape`, and with `dynamic_shape=True` it constructs
a model capable of processing an input image of any size.

This is an _open-data task_, where you submit only the test set annotations
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

Each test set image annotation consists of a sequence of space separated
five-tuples _label top left bottom right_, and the annotation is considered
correct, if exactly the gold digits are predicted, each with IoU at least 0.5.
The whole test set score is then the prediction accuracy of individual images.
An evaluation of a file with the predictions can be performed by the
[svhn_eval.py](https://github.com/ufal/npfl114/tree/master/labs/06/svhn_eval.py)
module.

The task is also a [_competition_](#competitions). Everyone submitting
a solution with at least _20%_ test set accuracy will get 5 points;
the rest 5 points will be distributed depending on relative ordering of your
solutions. Note that I usually need at least _35%_ development set accuracy
to achieve the required test set performance.

You should start with the
[svhn_competition.py](https://github.com/ufal/npfl114/tree/master/labs/06/svhn_competition.py)
template, which generates the test set annotation in the required format.

_A baseline solution can use RetinaNet-like single stage detector,
using only a single level of convolutional features (no FPN)
with single-scale and single-aspect anchors. Focal loss is available
as [tfa.losses.SigmoidFocalCrossEntropy](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy)
(using `reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE` option is a good
idea) and non-maximum suppression as
[tf.image.non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression) or
[tf.image.combined_non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression)._
