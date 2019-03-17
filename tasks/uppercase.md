### Assignment: uppercase
#### Date: Deadline: Mar 31, 23:59
#### Points: 4-9 points

This assignment introduces first NLP task. Your goal is to implement a model
which is given Czech lowercased text and tries to uppercase appropriate letters.
To load the dataset, use
[uppercase_data.py](https://github.com/ufal/npfl114/tree/master/labs/03/uppercase_data.py)
module which loads (and if required also downloads) the data. While the training
and the development sets are in correct case, the test set is lowercased.

This is an _open-data task_, where you submit only the uppercased test set
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.

The task is also a _competition_. Everyone who submits a solution which achieves
at least _96.5%_ accuracy will get 4 points; the rest 5 points will be distributed
depending on relative ordering of your solutions, i.e., the best solution will
get total 9 points, the worst solution (but at least with 96.5% accuracy) will
get total 4 points. The accuracy is computed per-character and can be evaluated
by [uppercase_eval.py](https://github.com/ufal/npfl114/tree/master/labs/03/uppercase_eval.py)
script.

You may want to start with the
[uppercase.py](https://github.com/ufal/npfl114/tree/master/labs/03/uppercase.py)
template, which uses the
[uppercase_data.py](https://github.com/ufal/npfl114/tree/master/labs/03/uppercase_data.py)
to load the data, generate an alphabet of given size containing most frequent
characters, and generate sliding window view on the data. The template also
comments on possibilities of character representation.

**Do not use RNNs or CNNs in this task, only densely connected layers (with
various activation and output functions).**
