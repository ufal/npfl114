### Assignment: tagger_we
#### Date: Deadline: Apr 26, 23:59
#### Points: 3 points
#### Examples: tagger_we_example

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

#### Examples Start: tagger_we_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 tagger_we.py --recodex --seed=7 --batch_size=2 --epochs=1 --threads=1 --max_sentences=200 --rnn_cell=LSTM --rnn_cell_dim=16 --we_dim=64`
  ```
  29.34
  ```
- `python3 tagger_we.py --recodex --seed=7 --batch_size=2 --epochs=1 --threads=1 --max_sentences=200 --rnn_cell=GRU --rnn_cell_dim=20 --we_dim=64`
  ```
  46.29
  ```
#### Examples End:
