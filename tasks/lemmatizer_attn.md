This task is a continuation of `lemmatizer_noattn` assignment.

Using the [lemmatizer_attn.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_attn.py)
template, add the following features in addition to `lemmatizer_noattn` ones:
- Run the encoder using bidirectional GRU.
- Implement attention in both decoders. Notably, project the encoder outputs and
  current state into same dimensionality vectors, apply non-linearity, and
  generate weights for every encoder output. Finally sum the encoder outputs
  using these weights and concatenate the computed attention to the decoder
  inputs.

Once submitted to ReCodEx, you should experiment with the effect of using
the attention, and the influence of RNN dimensionality on network performance.
