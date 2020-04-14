### Assignment: tagger_we
#### Date: Deadline: Apr 26, 23:59
#### Points: 3 points

In this assignment you will create a simple part-of-speech tagger. For training
and evaluation, we will use Czech dataset containing tokenized sentences, each
word annotated by gold lemma and part-of-speech tag. The
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/07/morpho_dataset.py)
module (down)loads the dataset and can generate batches.

Your goal is to modify the
[tagger_we.py](https://github.com/ufal/npfl114/tree/master/labs/07/tagger_we.py)
template and implement the following:
- Use specified RNN cell type (`GRU` and `LSTM`) and dimensionality.
- Create word embeddings for training vocabulary.
- Process the sentences using bidirectional RNN.
- Predict part-of-speech tags.
Note that you need to properly handle sentences of different lengths in one
batch using [masking](https://www.tensorflow.org/guide/keras/masking_and_padding).
