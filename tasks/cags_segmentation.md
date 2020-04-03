### Assignment: cags_segmentation
#### Date: Deadline: Apr 19, 23:59
#### Points: 6 points+5 bonus

The goal of this assignment is to use pretrained EfficientNet-B0 model to
achieve best image segmentation IoU score on the CAGS dataset.
The dataset and the EfficientNet-B0 is described in the `cags_classification`
assignment.

This is an _open-data task_, where you submit only the test set masks
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

A mask is evaluated using _intersection over union_ (IoU) metric, which is the
intersection of the gold and predicted mask divided by their union, and the
whole test set score is the average of its masks' IoU. A TensorFlow compatible
metric is implemented by the class `CAGSMaskIoU` of the
[cags_segmentation_eval.py](https://github.com/ufal/npfl114/tree/master/labs/05/cags_segmentation_eval.py)
module, which can further be used to evaluate a file with predicted masks.

The task is also a [_competition_](#competitions). Everyone who submits
a solution which achieves at least _85%_ test set IoU will get 6 points;
the rest 5 points will be distributed depending on relative ordering of your
solutions.

You may want to start with the
[cags_segmentation.py](https://github.com/ufal/npfl114/tree/master/labs/05/cags_segmentation.py)
template, which generates the test set annotation in the required format –
each mask should be encoded on a single line as a space separated sequence of
integers indicating the length of alternating runs of zeros and ones.
