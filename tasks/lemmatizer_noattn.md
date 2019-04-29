### Assignment: lemmatizer_noattn
#### Date: Deadline: May 12, 23:59
#### Points: 4 points

The goal of this assignment is to create a simple lemmatizer. For training
and evaluation, we will use Czech dataset containing tokenized sentences, each
word annotated by gold lemma and part-of-speech tag. The
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/07/morpho_dataset.py)
module (down)loads the dataset and can generate batches.

Your goal is to modify the
[lemmatizer_noattn.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_noattn.py)
template and implement the following:
- Embed characters of source forms and run a bidirectional GRU encoder.
- Embed characters of target lemmas.
- Implement a training time decoder which uses gold target characters as inputs.
- Implement an inference time decoder which uses previous predictions as inputs.
- The initial state of both decoders is the output state of the corresponding
  GRU encoded form.

After submitting the task to ReCodEx, continue with [`lemmatizer_attn`](#lemmatizer_attn) assignment.
