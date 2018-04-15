This assignment demonstrates usefulness of transfer learning. The goal is
to train a classifier for hand-drawn sketches. The dataset of 224×224
grayscale sketches categorized in 250 classes is available from
[nsketch.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nsketch.zip).
Again, this is an open data task, and your goal is to generate labels for
unannotated test set.

The task is a _competition_ and the points will be awarded depending on your
test set accuracy. If your test set accuracy surpasses 40%, you will be
awarded 6 points; the rest 6 points will be distributed depending on relative
ordering of your solutions.

To solve the task with transfer learning, start with a pre-trained ImageNet
network (NASNet A Mobile is used in the template, but feel free to use any)
and convert images to features. Then (probably in a separate script) train
a classifier processing the precomputed features into required classes.
_This approach leads to at least 52% accuracy on development set._
To improve the accuracy, you can then _finetune_ the original network – compose
the pre-trained ImageNet network together with the trained classifier and
continue training the whole composition. _Such finetuning should lead to at
least 70% accuracy on development set (using ResNet)._

You should start with the
[nsketch_transfer.py](https://github.com/ufal/npfl114/tree/master/labs/07/nsketch_transfer.py)
template, which loads the data, creates NASNet network and load its weight,
evaluates and predicts using batches, and on the end produces test set
annotations in the required format. However, feel free to use multiple scripts
for solving this assignment. The above template requires NASNet sources
and pretrained weights, which you can download among others
[here](http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip).
An independent example of using NASNet for classification is also available as
[nasnet_classify.py](https://github.com/ufal/npfl114/tree/master/labs/07/nasnet_classify.py).

To submit the test set annotations in ReCodEx, use the supplied
[nsketch_transfer_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/07/nsketch_transfer_recodex.py)
script. You need to provide at least two arguments – the first is the path to
the test set annotations and all other arguments are paths to the sources used
to generate the test data.
