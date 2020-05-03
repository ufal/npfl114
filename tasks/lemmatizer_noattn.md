### Assignment: lemmatizer_noattn
#### Date: Deadline: May 10, 23:59
#### Points: 4 points
#### Examples: lemmatizer_noattn_example

The goal of this assignment is to create a simple lemmatizer. For training
and evaluation, we use the same dataset as in `tagger_we` loadable by the
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/09/morpho_dataset.py)
module.

Your goal is to modify the
[lemmatizer_noattn.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_noattn.py)
template and implement the following:
- Embed characters of source forms and run a bidirectional GRU encoder.
- Embed characters of target lemmas.
- Implement a training time decoder which uses gold target characters as inputs.
- Implement an inference time decoder which uses previous predictions as inputs.
- The initial state of both decoders is the output state of the corresponding
  GRU encoded form.

#### Examples Start: lemmatizer_noattn_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 lemmatizer_noattn.py --recodex --seed=7 --batch_size=2 --epochs=3 --threads=1 --max_sentences=200 --rnn_dim=24 --cle_dim=64`
  ```
  20.47
  ```
#### Examples End:
