### Assignment: svhn_competition
#### Date: Deadline: Apr 4, 7:59
#### Points: 5 points+5 bonus

The goal of this assignment is to implement a system performing object
recognition, optionally utilizing pretrained EfficientNet-B0 backbone.

The [Street View House Numbers (SVHN) dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/demos/svhn_train.html)
annotates for every photo all digits appearing on it, including their bounding
boxes. The dataset can be loaded using the [svhn_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/06/svhn_dataset.py)
module. Similarly to the `CAGS` dataset, it is stored in a
[TFRecord file](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
with [tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
elements. Every element is a dictionary with the following keys:
- `"image"`: a square 3-channel image,
- `"classes"`: a 1D tensor with all digit labels appearing in the image,
- `"bboxes"`: a `[num_digits, 4]` 2D tensor with bounding boxes of every
  digit in the image.

Given that the dataset elements are each of possibly different size and you want
to preprocess them using `bboxes_training`, it might be more comfortable to
convert the dataset to NumPy. Alternatively, you can implement `bboxes_training`
using TensorFlow operations or call Numpy implementation of `bboxes_training`
directly in `tf.data.Dataset.map` by using `tf.numpy_function`,
see [FAQ](https://ufal.mff.cuni.cz/courses/npfl114/2122-summer#faq_tf_data).

Similarly to the `cags_classification`, you can load the EfficientNet-B0 using the provided
[efficient_net.py](https://github.com/ufal/npfl114/tree/master/labs/06/efficient_net.py)
module. Note that the `dynamic_input_shape=True` argument creates
a model capable of processing an input image of any size.

Each test set image annotation consists of a sequence of space separated
five-tuples _label top left bottom right_, and the annotation is considered
correct, if exactly the gold digits are predicted, each with IoU at least 0.5.
The whole test set score is then the prediction accuracy of individual images.
You can again evaluate your predictions using the
[svhn_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/06/svhn_dataset.py)
module, either by running with `--evaluate=path` arguments, or using its
`evaluate_file` method.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl114/2122-summer#competitions).
Everyone who submits a solution which achieves at least _20%_ test set IoU gets
5 points; the rest 5 points will be distributed depending on relative ordering
of your solutions. Note that I usually need at least _35%_ development set
accuracy to achieve the required test set performance.

You should start with the
[svhn_competition.py](https://github.com/ufal/npfl114/tree/master/labs/06/svhn_competition.py)
template, which generates the test set annotation in the required format.

_A baseline solution can use RetinaNet-like single stage detector,
using only a single level of convolutional features (no FPN)
with single-scale and single-aspect anchors. Focal loss is available
as [tf.losses.BinaryFocalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy)
and non-maximum suppression as
[tf.image.non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression) or
[tf.image.combined_non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression)._
