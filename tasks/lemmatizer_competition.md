### Assignment: lemmatizer_competition
#### Date: Deadline: May 17, 23:59
#### Points: 4 points+5 bonus

In this assignment, you should extend `lemmatizer_noattn` or `lemmatizer_attn`
into a real-world Czech lemmatizer. As in `tagger_competition`, we will use
Czech PDT dataset loadable using the [morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/10/morpho_dataset.py)
module.

You can also use the following additional data as in the `tagger_competition`
assignment.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl114/2021-summer#competitions). Everyone who submits a solution
a solution with at least 96% label accuracy gets 4 points; the rest 5 points
will be distributed depending on relative ordering of your solutions. Lastly,
**3 bonus points** will be given to anyone surpassing pre-neural-network
state-of-the-art of **97.86%**.

You can start with the
[lemmatizer_competition.py](https://github.com/ufal/npfl114/tree/master/labs/10/lemmatizer_competition.py)
template, which among others generates test set annotations in the required format. Note that
you can evaluate the predictions as usual using the [morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/10/morpho_dataset.py)
module, either by running with `--task=lemmatizer --evaluate=path` arguments, or using its
`evaluate_file` method.
