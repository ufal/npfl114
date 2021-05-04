### Assignment: lemmatizer_attn
#### Date: Deadline: May 17, 23:59
#### Points: 3 points
#### Examples: lemmatizer_attn_examples

This task is a continuation of the `lemmatizer_noattn` assignment. Using the
[lemmatizer_attn.py](https://github.com/ufal/npfl114/tree/master/labs/10/lemmatizer_attn.py)
template, implement the following features in addition to `lemmatizer_noattn`:
- The bidirectional GRU encoder returns outputs for all input characters, not
  just the last.
- Implement attention in the decoders. Notably, project the encoder outputs and
  current state into same dimensionality vectors, apply non-linearity, and
  generate weights for every encoder output. Finally sum the encoder outputs
  using these weights and concatenate the computed attention to the decoder
  inputs.

Once submitted to ReCodEx, you should experiment with the effect of using
the attention, and the influence of RNN dimensionality on network performance.

#### Examples Start: lemmatizer_attn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 lemmatizer_attn.py --max_sentences=1000 --batch_size=2 --cle_dim=24 --rnn_dim=24 --epochs=3`
```
Epoch 1/3 loss: 2.4224 - val_loss: 0.0000e+00 - val_accuracy: 0.1627
Epoch 2/3 loss: 1.8042 - val_loss: 0.0000e+00 - val_accuracy: 0.2574
Epoch 3/3 loss: 0.9277 - val_loss: 0.0000e+00 - val_accuracy: 0.2998
loss: 0.0000e+00 - accuracy: 0.3083
```
- `python3 lemmatizer_attn.py --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --epochs=3`
```
Epoch 1/3 loss: 2.6011 - val_loss: 0.0000e+00 - val_accuracy: 0.1232
Epoch 2/3 loss: 2.1855 - val_loss: 0.0000e+00 - val_accuracy: 0.2124
Epoch 3/3 loss: 1.4435 - val_loss: 0.0000e+00 - val_accuracy: 0.2649
loss: 0.0000e+00 - accuracy: 0.2815
```
#### Examples End:
