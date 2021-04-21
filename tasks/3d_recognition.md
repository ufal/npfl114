### Assignment: 3d_recognition
#### Date: Deadline: Apr 19, 23:59
#### Points: 5 points+5 bonus

Your goal in this assignment is to perform 3D object recognition. The input
is voxelized representation of an object, stored as a 3D grid of either empty
or occupied _voxels_, and your goal is to classify the object into one of
10 classes. The data is available in two resolutions, either as
[20×20×20 data](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/demos/modelnet20.html)
or [32×32×32 data](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/demos/modelnet32.html).
To load the dataset, use the
[modelnet.py](https://github.com/ufal/npfl114/tree/master/labs/06/modelnet.py) module.

The official dataset offers only train and test sets, with the **test set having
a different distributions of labels**. Our dataset contains also a development
set, which has **nearly the same** label distribution as the test set.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl114/2021-summer#competitions). Everyone who submits a solution
which achieves at least _87%_ test set accuracy gets 5 points; the rest
5 points will be distributed depending on relative ordering of your solutions.

You can start with the
[3d_recognition.py](https://github.com/ufal/npfl114/tree/master/labs/06/3d_recognition.py)
template, which among others generates test set annotations in the required format.
