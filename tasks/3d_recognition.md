### Assignment: 3d_recognition
#### Date: Deadline: Apr 26, 23:59
#### Points: 5 points+5 bonus

Your goal in this assignment is to perform 3D object recognition. The input
is voxelized representation of an object, stored as a 3D grid of either empty
or occupied _voxels_, and your goal is to classify the object into one of
10 classes. The data is available in two resolutions, either as
[20×20×20 data](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/demos/modelnet20.html)
or [32×32×32 data](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/demos/modelnet32.html).
To load the dataset, use the
[modelnet.py](https://github.com/ufal/npfl114/tree/master/labs/07/modelnet.py) module.

The official dataset offers only train and test sets, with the **test set having
a different distributions of labels**. Our dataset contains also a development
set, which has **nearly the same** label distribution as the test set.

The assignment is again an _open-data task_, where you submit only the test set labels
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

The task is also a [_competition_](#competitions). Everyone submitting
a solution with at least 85% test set accuracy will get 5 points; the rest
5 points will be distributed depending on relative ordering of your solutions.

You can start with the
[3d_recognition.py](https://github.com/ufal/npfl114/tree/master/labs/07/3d_recognition.py)
template, which among others generates test set annotations in the required format.
