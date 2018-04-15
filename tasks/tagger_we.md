In this assignment you will create a simple part-of-speech tagger.
For training and evaluation, use
[czech-cac.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/czech-cac.zip)
data containing Czech tokenized sentences, each word annotated by gold lemma
and part-of-speech tag. The dataset can be loaded using the
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_dataset.py)
module.

Your goal is to modify the
[tagger_we.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_we.py)
template and implement the following:
- Use specified RNN cell type (`GRU` and `LSTM`) and dimensionality.
- Create word embeddings for training vocabulary.
- Process the sentences using bidirectional RNN.
- Predict part-of-speech tags.
- You need to properly handle sentences of different lengths in one batch.
- Note how resettable metrics are handled by the template.

After submitting the task to ReCodEx, continue with `tagger_cle` and/or
`tagger_cnne` assignment.

You should also experiment with what effect does the RNN cell type and
cell dimensionality have on the results.
