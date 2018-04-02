Your goal in this assignment is to perform 3D object recognition. The input
is voxelized representation of an object, stored as a 3D grid of either empty
or occupied _voxels_, and your goal is to classify the object into one of
10 classes. The data is available in two resolutions, either as
[20×20×20 data](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/modelnet20.zip)
([visualization of objects of all classes](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/modelnet20.html))
or [32×32×32 data](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/modelnet32.zip)
([visualization of objects of all classes](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/modelnet32.html)).
As usual, this is an open data task; therefore, your goal is to generate
labels for unannotated test set. Note that the original dataset contains
only train and test portion – you need to use part of train portion as development set.

The task is a _competition_ and the points will be awarded depending on your
test set accuracy. If your test set score surpasses 75%, you will be
awarded 7 points; the rest 6 points will be distributed depending on relative
ordering of your solutions. _Note that quite a straightfoward model reaches
about 90% on the test set after several hours of training._

You should start with the
[3d_recognition.py](https://github.com/ufal/npfl114/tree/master/labs/06/3d_recognition.py)
template, which loads the data, split development set from the training data,
and on the end produces test set annotations in the required format.

To submit the test set annotations in ReCodEx, use the supplied
[3d_recognition_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/06/3d_recognition_recodex.py)
script. You need to provide at least two arguments -- the first is the path to
the test set annotations and all other arguments are paths to the sources used
to generate the test data.
