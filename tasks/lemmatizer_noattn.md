### Assignment: lemmatizer_noattn
#### Date: Deadline: May 2, 7:59 a.m.
#### Points: 3 points
#### Tests: lemmatizer_noattn_tests

The goal of this assignment is to create a simple lemmatizer. For training
and evaluation, we use the same dataset as in `tagger_we` loadable by the
updated [morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/10/morpho_dataset.py)
module.

Your goal is to modify the
[lemmatizer_noattn.py](https://github.com/ufal/npfl114/tree/master/labs/10/lemmatizer_noattn.py)
template and implement the following:
- Embed characters of source forms and run a bidirectional GRU encoder.
- Embed characters of target lemmas.
- Implement a training time decoder which uses gold target characters as inputs.
- Implement an inference time decoder which uses previous predictions as inputs.
- The initial state of both decoders is the output state of the corresponding
  GRU encoded form.
- If requested, tie the embeddings in the decoder.

#### Tests Start: lemmatizer_noattn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 lemmatizer_noattn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32`
```
250/250 - 15s - loss: 3.0551 - val_accuracy: 0.1196 - 15s/epoch - 59ms/step
```
2. `python3 lemmatizer_noattn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings`
```
250/250 - 15s - loss: 2.8971 - val_accuracy: 0.1409 - 15s/epoch - 58ms/step
```
#### Tests End:
