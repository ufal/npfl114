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
get total 6 points. The accuracy is computed per-character and will be evaluated
by [uppercase_eval.py](https://github.com/ufal/npfl114/tree/master/labs/03/uppercase_eval.py)
script.

If you want, you can start with the
[uppercase.py](https://github.com/ufal/npfl114/tree/master/labs/03/uppercase.py)
template, which loads the data, generate an alphabet of given size containing most frequent
characters, and can generate sliding window view on the data.
To represent characters, you might find `tf.one_hot` useful.

To submit the uppercased test set in ReCodEx, use the supplied
[uppercase_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/03/uppercase_recodex.py)
script. You need to provide at least two arguments -- the first is the path to
the uppercased test data and all other arguments are paths to the sources used
to generate the test data. Running the script will create
`uppercase_recodex_submission.py` file, which can be submitted in ReCodEx.

**Do not use RNNs or CNNs in this task, only densely connected layers (with
various activation and output functions).**
