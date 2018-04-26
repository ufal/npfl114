The goal of this task is to improve the state-of-the-art in Czech
lemmatization. The current state-of-the-art is (to my best knowledge)
[czech-morfflex-pdt-161115](http://ufal.mff.cuni.cz/morphodita/users-manual#czech-morfflex-pdt_model)
reimplementation of [Spoustová et al., 2009](http://www.aclweb.org/anthology/E09-1087)
tagger and achieves 97.86% lemma accuracy.

As in `tagger_sota` assignment, for training use the
[czech-pdt.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/czech-pdt.zip)
dataset, which can be loaded employing the
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_dataset.py)
module. Additionally, you can also use outputs of a morphological analyzer
[czech-pdt-analysis.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/czech-pdt-analysis.zip).

This task is an open-data competition and the points will be awarded depending on your
test set accuracy. If your test set accuracy surpasses 90%, you will be
awarded 4 points; the rest 4 points will be distributed depending on relative
ordering of your solutions. Any solution surpassing 97.86% will get additional 5 points.
The evaluation (using for example development data) can be performed by
[morpho_eval.py](https://github.com/ufal/npfl114/tree/master/labs/09/morpho_eval.py)
script.

You can start with the
[lemmatizer_sota.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_sota.py)
template, which loads the PDT data, loads the morphological analysers data, and
finally generates the predictions in the required format (which is exactly the
same as the input format).

To submit the test set annotations in ReCodEx, use the supplied
[lemmatizer_sota_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_sota_recodex.py)
script. You need to provide at least two arguments – the first is the path to
the test set annotations and all other arguments are paths to the sources used
to generate the test data.
