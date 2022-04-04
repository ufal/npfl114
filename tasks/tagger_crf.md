### Assignment: tagger_crf
#### Date: Deadline: Apr 19, 7:59 a.m.
#### Points: 2 points

This assignment is an extension of `tagger_we` task. Using the
[tagger_crf.py](https://github.com/ufal/npfl114/tree/master/labs/08/tagger_crf.py)
template, implement named entity recognition using CRF loss and CRF decoding
from the `tensorflow_addons` package.

The evaluation is performed using the provided metric computing F1 score of the
span prediction (i.e., a recognized possibly-multiword named entity is true
positive if both the entity type and the span exactly match).

In practice, character-level embeddings (and also pre-trained word embeddings)
would be used to obtain superior results.

