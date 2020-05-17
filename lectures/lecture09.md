### Lecture: 9. Word2Vec, Seq2seq, NMT
#### Date: Apr 27
#### Slides: https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/slides/?09
#### Reading: https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/slides.pdf/npfl114-09.pdf,PDF Slides
#### Video: https://lectures.ms.mff.cuni.cz/video/rec/npfl114/1920/npfl114-09.mp4,Video
#### Video: https://lectures.ms.mff.cuni.cz/video/rec/npfl114/1920/npfl114-09.practicals.mp4,Video – Practicals
#### Video: https://slideslive.com/38907189/deep-learning-lecture-8-recurrent-neural-networks-ii-word-embeddings, 2018 Video I
#### Video: https://slideslive.com/38907422/deep-learning-lecture-9-recurrent-neural-networks-iii-machine-translation, 2018 Video II
#### Questions: #lecture_999999999_questions
#### Lecture assignment: tensorboard_projector
#### Lecture assignment: lemmatizer_noattn
#### Lecture assignment: lemmatizer_attn
#### Lecture assignment: lemmatizer_competition
#### VideoPlayer: npfl114-09.mp4,npfl114-09.practicals.mp4

- `Word2vec` word embeddings, notably the CBOW and Skip-gram architectures [[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)]
  - Hierarchical softmax [Section 12.4.3.2 of DLB or [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)]
  - Negative sampling [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)]
- _Character-level embeddings using character n-grams [Described simultaneously in several papers as Charagram ([Charagram: Embedding Words and Sentences via Character n-grams](https://arxiv.org/abs/1607.02789)), Subword Information ([Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) or SubGram ([SubGram: Extending Skip-Gram Word Representation with Substrings](http://link.springer.com/chapter/10.1007/978-3-319-45510-5_21))]_
- Neural Machine Translation using Encoder-Decoder or Sequence-to-Sequence architecture [Section 12.5.4 of DLB, [Ilya Sutskever, Oriol Vinyals, Quoc V. Le: **Sequence to Sequence Learning with Neural Networks**](https://arxiv.org/abs/1409.3215) and [Kyunghyun Cho et al.: **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation**](https://arxiv.org/abs/1406.1078)]
- Using Attention mechanism in Neural Machine Translation [Section 12.4.5.1 of DLB, [Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio: **Neural Machine Translation by Jointly Learning to Align and Translate**](https://arxiv.org/abs/1409.0473)]
- Translating Subword Units [[Rico Sennrich, Barry Haddow, Alexandra Birch: **Neural Machine Translation of Rare Words with Subword Units**](https://arxiv.org/abs/1508.07909)]
- _Google NMT [[Yonghui Wu et al.: Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)]_
