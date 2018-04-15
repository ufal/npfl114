The goal of this assignment is to devise the best possible model for MNIST
data set. However, in order for the test set results not to be available,
use the data from
[mnist-gan.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan.zip).
It was created using GANs (generative adversarial networks) from the original
MNIST data and contain fake test labels (all labels are 255).

This is an _open-data task_, you will submit only test set labels (in addition
to a training script, which will be used only to understand the approach you
took).

The task is a _competition_ and the points will be awarded depending on your
test set accuracy. If your test set accuracy surpasses 99.4%, you will be
awarded 5 points; the rest 5 points will be distributed depending on relative
ordering of your solutions.

The
[mnist_competition.py](https://github.com/ufal/npfl114/tree/master/labs/04/mnist_competition.py)
template loads data from `mnist-gan` directory and in the end saves
the test labels in the required format (each label on a separate line).

To submit the test set labels in ReCodEx, use the supplied
[mnist_competition_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/04/mnist_competition_recodex.py)
script. You need to provide at least two arguments – the first is the path to
the test set labels and all other arguments are paths to the sources used
to generate the test data. Running the script will create
`mnist_competition_recodex_submission.py` file, which can be submitted in ReCodEx.
