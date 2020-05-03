### Assignment: lemmatizer_competition
#### Date: Deadline: ~~May 10, 23:59~~ May 17, 23:59
#### Points: 5 points+5 bonus

In this assignment, you should extend `lemmatizer_noattn` or `lemmatizer_attn`
into a real-world Czech lemmatizer. As in `tagger_competition`, we will use
Czech PDT dataset loadable using the [morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/09/morpho_dataset.py)
module.

You can also use the following additional data as in the `tagger_competition`
assignment.

The assignment is again an _open-data task_, where you submit only the test set annotations
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.
**Note that all `.zip` files you submit will be extracted first.**

The task is also a [_competition_](#competitions). Everyone submitting
a solution with at least 92% accuracy will get 5 points; the rest 5 points will be distributed
depending on relative ordering of your solutions. Lastly, **3 bonus points**
will be given to anyone surpassing pre-neural-network state-of-the-art
of 97.86%. You can evaluate a generated file using the
[morpho_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/09/morpho_evaluator.py)
module.

You can start with the
[lemmatizer_competition.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_competition.py)
template, which among others generates test set annotations in the required format.
