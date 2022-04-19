### Assignment: 3d_recognition
#### Date: Deadline: May 02, 7:59 a.m.
#### Points: 3 points+4 bonus

Your goal in this assignment is to perform 3D object recognition. The input
is voxelized representation of an object, stored as a 3D grid of either empty
or occupied _voxels_, and your goal is to classify the object into one of
10 classes. The data is available in two resolutions, either as
[20×20×20 data](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/demos/modelnet20.html)
or [32×32×32 data](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/demos/modelnet32.html).
To load the dataset, use the
[modelnet.py](https://github.com/ufal/npfl114/tree/master/labs/10/modelnet.py) module.

The official dataset offers only train and test sets, with the **test set having
a different distributions of labels**. Our dataset contains also a development
set, which has **nearly the same** label distribution as the test set.

If you want, it is possible to use the EfficientNet-B0 in this assignment;
however, I do not know of a straightforward way to utilize it, apart from
rendering the object to a 2D image (or several of them).

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl114/2122-summer#competitions). Everyone who submits a solution
which achieves at least _88%_ test set accuracy gets 3 points; the rest
4 points will be distributed depending on relative ordering of your solutions.

You can start with the
[3d_recognition.py](https://github.com/ufal/npfl114/tree/master/labs/10/3d_recognition.py)
template, which among others generates test set annotations in the required format.
