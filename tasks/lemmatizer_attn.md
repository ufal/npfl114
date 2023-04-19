### Assignment: lemmatizer_attn
#### Date: Deadline: May 2, 7:59 a.m.
#### Points: 3 points
#### Tests: lemmatizer_attn_tests
#### Examples: lemmatizer_attn_examples

This task is a continuation of the `lemmatizer_noattn` assignment. Using the
[lemmatizer_attn.py](https://github.com/ufal/npfl114/tree/master/labs/10/lemmatizer_attn.py)
template, implement the following features in addition to `lemmatizer_noattn`:
- The bidirectional GRU encoder returns outputs for all input characters, not
  just the last.
- Implement attention in the decoders. Notably, project the encoder outputs and
  current state into same-dimensionality vectors, apply non-linearity, and
  generate weights for every encoder output. Finally sum the encoder outputs
  using these weights and concatenate the computed attention to the decoder
  inputs.

Once submitted to ReCodEx, you should experiment with the effect of using
the attention, and the influence of RNN dimensionality on network performance.

#### Tests Start: lemmatizer_attn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 lemmatizer_attn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32`
```
loss: 3.0485 - val_accuracy: 0.0794 - 22s/epoch - 89ms/step
```
2. `python3 lemmatizer_attn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings`
```
loss: 2.8510 - val_accuracy: 0.1601 - 22s/epoch - 88ms/step
```
#### Tests End:
#### Examples Start: lemmatizer_attn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 lemmatizer_attn.py --epochs=3 --max_sentences=5000`
```
Epoch 1/3 loss: 1.8517 - val_accuracy: 0.6009 - 62s/epoch - 125ms/step
Epoch 2/3 loss: 0.3652 - val_accuracy: 0.7027 - 48s/epoch - 95ms/step
Epoch 3/3 loss: 0.2269 - val_accuracy: 0.7944 - 48s/epoch - 96ms/step
```
- `python3 lemmatizer_attn.py --epochs=3 --max_sentences=5000 --tie_embeddings`
```
Epoch 1/3 loss: 1.6228 - val_accuracy: 0.6352 - 61s/epoch - 122ms/step
Epoch 2/3 loss: 0.3158 - val_accuracy: 0.7529 - 48s/epoch - 96ms/step
Epoch 3/3 loss: 0.1949 - val_accuracy: 0.7913 - 48s/epoch - 96ms/step
```
#### Examples End:
