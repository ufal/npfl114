### Assignment: tagger_competition
#### Date: Deadline: May 03, 23:59
#### Points: 4 points+5 bonus

In this assignment, you should extend
`tagger_we`/`tagger_cle_rnn`/`tagger_cle_cnn`
into a real-world Czech part-of-speech tagger. We will use
Czech PDT dataset loadable using the [morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_dataset.py)
module. Note that the dataset contains more than 1500 unique POS tags and that
the POS tags have a fixed structure of 15 positions (so it is possible to
generate the POS tag characters independently).

You can use the following additional data in this assignment:
- You can use outputs of a morphological analyzer loadable with
  [morpho_analyzer.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_analyzer.py).
  If a word form in train, dev or test PDT data is known to the analyzer,
  all its _(lemma, POS tag)_ pairs are returned.
- You can use any _unannotated_ text data (Wikipedia, Czech National Corpus, …).

The assignment is again an _open-data task_, where you submit only the test set annotations
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.
**Note that all `.zip` files you submit will be extracted first.**

The task is also a [_competition_](#competitions). Everyone submitting
a solution with at least 92% label accuracy gets 4 points; the rest 5 points
will be distributed depending on relative ordering of your solutions. Lastly,
**3 bonus points** will be given to anyone surpassing pre-neural-network
state-of-the-art of 95.89% from [Spoustová et al., 2009](http://www.aclweb.org/anthology/E09-1087).
You can evaluate a generated file using the
[morpho_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_evaluator.py)
module.

You can start with the
[tagger_competition.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_competition.py)
template, which among others generates test set annotations in the required format.
