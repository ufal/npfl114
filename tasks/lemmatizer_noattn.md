### Assignment: lemmatizer_noattn
#### Date: Deadline: Apr 25, 7:59 a.m.
#### Points: 3 points
#### Tests: lemmatizer_noattn_tests

The goal of this assignment is to create a simple lemmatizer. For training
and evaluation, we use the same dataset as in `tagger_we` loadable by the
updated [morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/09/morpho_dataset.py)
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

#### Tests Start: lemmatizer_noattn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 lemmatizer_noattn.py --epochs=1 --max_sentences=1000 --batch_size=2 --cle_dim=24 --rnn_dim=24`
```
500/500 - 29s - loss: 2.9655 - val_accuracy: 0.1311 - 29s/epoch - 58ms/step
```
- `python3 lemmatizer_noattn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32`
```
250/250 - 17s - loss: 3.0641 - val_accuracy: 0.0043 - 17s/epoch - 69ms/step
```
#### Tests End:
