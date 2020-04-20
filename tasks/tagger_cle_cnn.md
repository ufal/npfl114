### Assignment: tagger_cle_cnn
#### Date: Deadline: May 03, 23:59
#### Points: 2 points

This task is a continuation of `tagger_cle_rnn` assignment.

Instead of using RNNs to generate character-level embeddings, process embedded
unique words with 1D convolutional filters with kernel sizes of 2 to some given
maximum. To obtain a fixed-size representation, perform global max-pooling over
the whole word.
