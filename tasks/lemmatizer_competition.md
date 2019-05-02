### Assignment: lemmatizer_competition
#### Date: Deadline: May 12, 23:59
#### Points: 5-13 points

In this assignment, you should extend
[`lemmatizer_noattn`](#lemmatizer_noattn)/[`lemmatizer_attn`](#lemmatizer_attn)
into a real-world Czech lemmatizer. We will again use
Czech PDT dataset loadable using the [morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/09/morpho_dataset.py)
module.

You can use the following additional data in this assignment:
- You can use outputs of a morphological analyzer loadable with
  [morpho_analyzer.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_analyzer.py).
  If a word form in train, dev or test PDT data is known to the analyzer,
  all its _(lemma, POS tag)_ pairs are returned.
- You can use any _unannotated_ text data (Wikipedia, Czech National Corpus, …).

The assignment is again an _open-data task_, where you submit only the annotated test set
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.
**Note that all `.zip` files you submit will be extracted first.**

The task is also a _competition_. Everyone who submits a solution which achieves
at least 92% accuracy will get 5 points; the rest 5 points will be distributed
depending on relative ordering of your solutions. Lastly, **3 bonus points**
will be given to anyone surpassing pre-neural-network state-of-the-art
of 97.86%. You can evaluate generated file against a golden text file using the
[morpho_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_evaluator.py)
module.

You can start with the
[lemmatizer_competition.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_competition.py)
template, which among others generates test set annotations in the required format.
