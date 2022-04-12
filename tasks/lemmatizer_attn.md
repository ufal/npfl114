### Assignment: lemmatizer_attn
#### Date: Deadline: Apr 25, 7:59 a.m.
#### Points: 3 points
#### Tests: lemmatizer_attn_tests

This task is a continuation of the `lemmatizer_noattn` assignment. Using the
[lemmatizer_attn.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_attn.py)
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

#### Tests Start: lemmatizer_attn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 lemmatizer_attn.py --epochs=1 --max_sentences=1000 --batch_size=2 --cle_dim=24 --rnn_dim=24`
```
500/500 - 33s - loss: 2.8881 - val_accuracy: 0.1451 - 33s/epoch - 66ms/step
```
- `python3 lemmatizer_attn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32`
```
250/250 - 29s - loss: 3.0441 - val_accuracy: 0.1471 - 29s/epoch - 114ms/step
```
#### Tests End:
