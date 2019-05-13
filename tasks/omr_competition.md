### Assignment: omr_competition
#### Date: Deadline: May 26, 23:59
#### Points: 7-15 points

You should implement optical music recognition in your final competition
assignment. The inputs are PNG images of monophonic scores starting with
a clef, key signature, and a time signature, followed by several staves.
The [dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/demos/omr_train.html)
is loadable using the [omr_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/11/omr_dataset.py)
module and is downloaded automatically if missing (note that is has 185MB).
No other data or pretrained models are allowed for training.

The assignment is again an _open-data task_, where you submit only the annotated test set
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.
**Note that all `.zip` files you submit will be extracted first.**

The task is also a _competition_. The evaluation is performed by computing edit
distance to the gold mark sequence, normalized by its length (i.e., exactly as
`tf.edit_distance`). Everyone who submits a solution which achieves
at most _10%_ test set edit distance will get 7 points; the rest 4 points will be distributed
depending on relative ordering of your solutions. Furthermore, **4 bonus points**
will be given to anyone surpassing current state-of-the-art of 0.80%.
An evaluation (using for example development data) can be performed by
[speech_recognition_eval.py](https://github.com/ufal/npfl114/tree/master/labs/07/speech_recognition_eval.py).

You can start with the
[omr_competition.py](https://github.com/ufal/npfl114/tree/master/labs/11/omr_competition.py)
template, which among others generates test set annotations in the required format.
