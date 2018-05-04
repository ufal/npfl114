This task is an extension of `tagger_we` assignment.

Using the [tagger_crf.py](https://github.com/ufal/npfl114/tree/master/labs/11/tagger_crf.py)
template, in addition to `tagger_we` features, implement training and decoding
with a CRF output layer, using the `tf.contrib.crf` module.

Once submitted to ReCodEx, you should experiment with the effect of CRF
compared to plain `tagger_we`. Note however that the effect of CRF on tagging
is minor – more appropriate task is for example named entity recognition,
which you can experiment with using Czech Named Entity Corpus
[czech-cnec.zip](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/czech-cnec.zip).
