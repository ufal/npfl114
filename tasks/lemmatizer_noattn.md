### Assignment: lemmatizer_noattn
#### Date: Deadline: May 17, 23:59
#### Points: 3 points
#### Examples: lemmatizer_noattn_examples

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

#### Examples Start: lemmatizer_noattn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 lemmatizer_noattn.py --max_sentences=1000 --batch_size=2 --cle_dim=24 --rnn_dim=24 --epochs=3`
```
Epoch 1/3 loss: 2.5645 - val_loss: 0.0000e+00 - val_accuracy: 0.1372
Epoch 2/3 loss: 1.9879 - val_loss: 0.0000e+00 - val_accuracy: 0.2061
Epoch 3/3 loss: 1.4119 - val_loss: 0.0000e+00 - val_accuracy: 0.2874
loss: 0.0000e+00 - accuracy: 0.2921
```
- `python3 lemmatizer_noattn.py --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --epochs=3`
```
Epoch 1/3 loss: 2.5907 - val_loss: 0.0000e+00 - val_accuracy: 0.1206
Epoch 2/3 loss: 2.1792 - val_loss: 0.0000e+00 - val_accuracy: 0.2160
Epoch 3/3 loss: 1.5338 - val_loss: 0.0000e+00 - val_accuracy: 0.2590
loss: 0.0000e+00 - accuracy: 0.2653
```
#### Examples End:
