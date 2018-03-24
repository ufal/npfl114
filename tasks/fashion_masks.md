This assignment is a simple image segmentation task. The data for this task is
available from [fashion-masks.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/fashion-masks.zip).
The inputs consist of 28×28 greyscale images of ten classes of clothing,
while the outputs consist of the correct class _and_ a pixel bit mask.
Your goal is to generate such outputs for the test set (including
to a training script, which will be used only to understand the approach you
took).

Performance is evaluated using mean IoU, where IoU for a single example
is defined as an intersection of the gold and system mask divided by
their union (assuming the predicted label is correct; if not, IoU is 0).
The evaluation (using for example development data) can be performed by
[fashion_masks_eval.py](https://github.com/ufal/npfl114/tree/master/labs/05/fashion_masks_eval.py)
script.

The task is a _competition_ and the points will be awarded depending on your
test set score. If your test set score surpasses 75%, you will be
awarded 6 points; the rest 6 points will be distributed depending on relative
ordering of your solutions. _Note that quite a straightfoward model surpasses
80% on development set after an hour of computation (and 90% after several
hours), so reaching 75% is not that difficult._

You should start with the
[fashion_masks.py](https://github.com/ufal/npfl114/tree/master/labs/05/fashion_masks.py)
template, which loads the data, computes averate IoU and on the end produces
test set annotations in the required format (one example per line containing
space separated label and mask, the mask stored as zeros and ones, rows first).

To submit the test set annotations in ReCodEx, use the supplied
[fashion_masks_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/05/fashion_masks_recodex.py)
script. You need to provide at least two arguments -- the first is the path to
the test set annotations and all other arguments are paths to the sources used
to generate the test data.
