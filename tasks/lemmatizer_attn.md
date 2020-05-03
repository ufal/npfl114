### Assignment: lemmatizer_attn
#### Date: Deadline: May 10, 23:59
#### Points: 3 points
#### Examples: lemmatizer_attn_example

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

#### Examples Start: lemmatizer_attn_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 lemmatizer_attn.py --recodex --seed=7 --batch_size=2 --epochs=3 --threads=1 --max_sentences=200 --rnn_dim=24 --cle_dim=64`
  ```
  22.14
  ```
#### Examples End:
