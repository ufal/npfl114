### Assignment: caltech42_competition
#### Date: Deadline: Apr 21, 23:59
#### Points: 5-10 points

The goal of this assignment is to try transfer learning approach to train image
recognition on a small dataset with 42 classes. You can load the data using the
[caltech42.py](https://github.com/ufal/npfl114/tree/master/labs/06/caltech42.py)
module. In addition to the training data, you should use a MobileNet v2
pretrained network (details in [caltech42_competition.py](https://github.com/ufal/npfl114/tree/master/labs/06/caltech42_competition.py)).

This is an _open-data task_, where you submit only the test set labels
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

The task is also a _competition_. Everyone who submits a solution which achieves
at least _94%_ test set accuracy will get 5 points; the rest 5 points will be distributed
depending on relative ordering of your solutions.

You may want to start with the
[caltech42_competition.py](https://github.com/ufal/npfl114/tree/master/labs/06/caltech42_competition.py)
template.
