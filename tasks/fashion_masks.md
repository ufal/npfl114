### Assignment: fashion_masks
#### Date: Deadline: Apr 14, 23:59
#### Points: 5-11 points

**The assignment is not yet in ReCodEx, it will appear there soon.**

This assignment is a simple image segmentation task. The data for this task is
available through the [fashion_masks_data.py](https://github.com/ufal/npfl114/tree/master/labs/05/fashion_masks_data.py)
The inputs consist of 28×28 greyscale images of ten classes of clothing,
while the outputs consist of the correct class _and_ a pixel bit mask.

This is an _open-data task_, where you submit only the test set annotations
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

Performance is evaluated using mean IoU, where IoU for a single example
is defined as an intersection of the gold and system mask divided by
their union (assuming the predicted label is correct; if not, IoU is 0).
The evaluation (using for example development data) can be performed by
[fashion_masks_eval.py](https://github.com/ufal/npfl114/tree/master/labs/05/fashion_masks_eval.py)
script.

The task is a _competition_ and the points will be awarded depending on your
test set score. If your test set score surpasses 75%, you will be
awarded 5 points; the rest 6 points will be distributed depending on relative
ordering of your solutions. _Note that quite a straightfoward model surpasses
80% on development set after an hour of computation (and 90% after several
hours), so reaching 75% is not that difficult._

You may want to start with the
[fashion_masks.py](https://github.com/ufal/npfl114/tree/master/labs/05/fashion_masks.py)
template, which loads the data and generates test set annotations in the
required format (one example per line containing space separated label and mask,
the mask stored as zeros and ones, rows first).
