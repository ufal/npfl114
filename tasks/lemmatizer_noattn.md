In this assignment you will create a simple lemmatizer.
For training and evaluation, use
[czech-cac.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/czech-cac.zip)
data containing Czech tokenized sentences, each word annotated by gold lemma
and part-of-speech tag. The dataset can be loaded using the
[morpho_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/08/morpho_dataset.py)
module.

Your goal is to modify the
[lemmatizer_noattn.py](https://github.com/ufal/npfl114/tree/master/labs/09/lemmatizer_noattn.py)
template and implement the following:
- Embed characters of source forms and run a forward GRU encoder.
- Embed characters of target lemmas.
- Implement a training time decoder which uses gold target characters as inputs.
- Implement an inference time decoder which uses previous predictions as inputs.
- The initial state of both decoders is the output state of the corresponding
  GRU encoded form.

After submitting the task to ReCodEx, continue with `lemmatizer_attn` assignment.
