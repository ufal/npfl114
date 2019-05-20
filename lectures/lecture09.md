### Lecture: 9. Recurrent Neural Networks III
#### Date: Apr 29
#### Slides: https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/slides/?09
#### Reading: https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/slides.pdf/npfl114-09.pdf,PDF Slides
#### Video: https://slideslive.com/38907189/deep-learning-lecture-8-recurrent-neural-networks-ii-word-embeddings, 2018 Video I
#### Video: https://slideslive.com/38907422/deep-learning-lecture-9-recurrent-neural-networks-iii-machine-translation, 2018 Video II
#### Video: https://slideslive.com/38907562/deep-learning-lecture-11-sequence-prediction-reinforcement-learning, 2018 Video III
#### Video: https://slideslive.com/38910718/deep-learning-lecture-12-sequence-prediction-ii-reinforcement-learning-ii, 2018 Video IV
#### Lecture assignment: lemmatizer_noattn
#### Lecture assignment: lemmatizer_attn
#### Lecture assignment: lemmatizer_competition

- Connectionist Temporal Classification (CTC) loss [[A. Graves, S. Fernández, F. Gomez, J. Schmidhuber: **Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks**](https://www.cs.toronto.edu/~graves/icml_2006.pdf)]
- `Word2vec` word embeddings, notably the CBOW and Skip-gram architectures [[Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean: Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)]
  - Hierarchical softmax [Section 12.4.3.2 of DLB or [Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean: Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)]
  - Negative sampling [Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean: Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)]
- Neural Machine Translation using Encoder-Decoder or Sequence-to-Sequence architecture [Section 12.5.4 of DLB, [Ilya Sutskever, Oriol Vinyals, Quoc V. Le: **Sequence to Sequence Learning with Neural Networks**](https://arxiv.org/abs/1409.3215) and [Kyunghyun Cho et al.: **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation**](https://arxiv.org/abs/1406.1078)]
- Using Attention mechanism in Neural Machine Translation [Section 12.4.5.1 of DLB, [Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio: **Neural Machine Translation by Jointly Learning to Align and Translate**](https://arxiv.org/abs/1409.0473)]
- Translating Subword Units [[Rico Sennrich, Barry Haddow, Alexandra Birch: **Neural Machine Translation of Rare Words with Subword Units**](https://arxiv.org/abs/1508.07909)]
- _Google NMT [[Yonghui Wu et al.: Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)]_
