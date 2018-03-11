This assignment introduces first textual task. Your goal is to implement
a network which is given a Czech text and it tries to uppercase appropriate
letters. Specifically, your goal is to uppercase given test set as well as
possible. The task data is available in
[uppercase_data.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/uppercase_data.zip)
archive. While the training and the development sets are in correct case, the
test set is all in lowercase.

This is an _open-data task_, so you will submit only the uppercased test set
(in addition to a training script, which will be used only to understand the
approach you took).

The task is also a _competition_. Everyone who submits a solution which achieves
at least 96.5% accuracy will get 6 points; the rest 4 points will be distributed
depending on relative ordering of your solutions, i.e., the best solution will
get total 10 points, the worst solution (but at least with 96.5% accuracy) will
get total 6 points.

If you want, you can start with the
[uppercase.py](https://github.com/ufal/npfl114/tree/master/labs/03/uppercase.py)
template, which loads the data, generate an alphabet of given size containing most frequent
characters, and can generate sliding window view on the data.
To represent characters, you might find `tf.one_hot` useful.

**The task will be possible to submit in ReCodEx next week. Note that you
will have only 5 submission attempts.**

**Do not use RNNs or CNNs in this task, only densely connected layers (with
various activation and output functions).**

