In this competition you will be solving the Native Language Identification
task. In that task, you get an English essay writen by a non-native individual
and your goal is to identify their native language.

We will be using NLI Shared Task 2013 data, which contains documents in 11
languages. For each language, the train, development and test sets contain
900, 100 and 100 documents, respectively. Particularly interesting is the fact
that humans are very bad in this task, while machine learning models can achive
quite high accuracy. Notably, the 2013 shared tasks winners achieved 83.6%
accuracy, while current state-of-the-art is at least 87.1% (Malmasi and Dras,
2017).

Because the data is not publicly available, you can download it only through
ReCodEx. Please do not distribute it. To load the dataset, you can use
[nli_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/10/nli_dataset.py)
script.

This task is an open-data competition and the points will be awarded depending on your
test set accuracy. If your test set accuracy surpasses 50%, you will be
awarded 6 points; the rest 6 points will be distributed depending on relative
ordering of your solutions. An evaluation (using for example development data)
can be performed by [nli_eval.py](https://github.com/ufal/npfl114/tree/master/labs/10/nli_eval.py).

You can start with the
[nli.py](https://github.com/ufal/npfl114/tree/master/labs/10/nli.py)
template, which loads the data and generates predictions in the required format
(language of each essay on a line).

To submit the test set annotations in ReCodEx, use the supplied
[nli_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/10/nli_recodex.py)
script. You need to provide at least two arguments – the first is the path to
the test set annotations and all other arguments are paths to the sources used
to generate the test data.
