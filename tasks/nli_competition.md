### Assignment: nli_competition
#### Date: Deadline: May 19, 23:59
#### Points: 6-10 points

In this competition you will be solving the Native Language Identification
task. In that task, you get an English essay writen by a non-native individual
and your goal is to identify their native language.

We will be using NLI Shared Task 2013 data, which contains documents in 11
languages. For each language, the train, development and test sets contain
900, 100 and 100 documents, respectively. Particularly interesting is the fact
that humans are quite bad in this task (in a simplified settings, human professionals
achieve 40-50% accuracy), while machine learning models can achive
high performance. Notably, the 2013 shared tasks winners achieved 83.6%
accuracy, while current state-of-the-art is at least 87.1% (Malmasi and Dras,
2017).

Because the data is not publicly available, you can download it only through
ReCodEx. Please do not distribute it. To load the dataset, use
[nli_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/10/nli_dataset.py)
script.

The assignment is again an _open-data task_, where you submit only the annotated test set
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.
**Note that all `.zip` files you submit will be extracted first.**

The task is also a _competition_. If your test set accuracy surpasses 60%, you will be
awarded 6 points; the rest 4 points will be distributed depending on relative
ordering of your solutions.

You can start with the
[nli_competition.py](https://github.com/ufal/npfl114/tree/master/labs/10/nli_competition.py)
template, which loads the data and generates predictions in the required format
(language of each essay on a line).
