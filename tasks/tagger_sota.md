The goal of this task is to improve the state-of-the-art in Czech
part-of-speech tagging. The current state-of-the-art is (to my best knowledge)
from [Spoustová et al., 2009](http://www.aclweb.org/anthology/E09-1087)
and is 95.67% in supervised and 95.89% in semi-supervised settings.

For training use the
[czech-pdt.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/czech-pdt.zip)
dataset, which can be loaded using the
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_dataset.py)
module. Note that the dataset contains more than 1500 unique POS tags and that
the POS tags have a fixed structure of 15 positions (so it is possible to
generate the POS tag characters independently).

Additionally, you can also use outputs of a morphological analyzer
[czech-pdt-analysis.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/czech-pdt-analysis.zip).
For each word form in train, dev and test PDT data, an analysis is present
either in a file containing results from a manually generated morphological
dictionary, or in a file with results from a trained morphological guesser.
Both files have the same structure – each line describes one word form which is
stored on the beginning of the line, followed by tab-separated lemma-tag pairs
from the analyzer.

This task is an open-data competition and the points will be awarded depending on your
test set accuracy. If your test set accuracy surpasses 90%, you will be
awarded 4 points; the rest 6 points will be distributed depending on relative
ordering of your solutions. Any solution surpassing 95.89% will get additional 5 points.
The evaluation (using for example development data) can be performed by
[morpho_eval.py](https://github.com/ufal/npfl114/tree/master/labs/09/morpho_eval.py)
script.

You can start with the
[tagger_sota.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_sota.py)
template, which loads the PDT data, loads the morphological analysers data, and
finally generates the predictions in the required format (which is exactly the
same as the input format).

To submit the test set annotations in ReCodEx, use the supplied
[tagger_sota_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_sota_recodex.py)
script. You need to provide at least two arguments – the first is the path to
the test set annotations and all other arguments are paths to the sources used
to generate the test data.
