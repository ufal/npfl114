Generally, only the topics covered on the lecture are part of the exam
(i.e., you should be able to tell me what I told you). The references
are to Deep Learning Book, unless stated otherwise.

- Computation model of neural networks
    - acyclic graph with nodes and edges
    - evaluation (forward propagation) _[Algorithm 6.1]_
    - activation functions _[`tanh` and `ReLU`s, including equations]_
    - output functions _[`σ` and `softmax`, including equations (3.30 and 6.29);
      you should also know how `softmax` is implemented to avoid overflows]_

- Backpropagation algorithm *[Algorithm 6.2; Algorithms 6.5 and 6.6 are used in practise,
  i.e., during `tf.train.Optimizer.compute_gradients`, so you should understand the idea
  behind them, but you do not have to understand the notation of `op.bprop` etc. from
  Algorithms 6.5 and 6.6]*

- Gradient descent and stochastic gradient descent algorithm _[Section 5.9]_

- Maximum likelihood estimation (MLE) principle _[Section 5.5, excluding 5.5.2]_
    - negative log likelihood as a loss derived by MLE
    - mean square error loss derived by MLE from Gaussian prior _[Equations (5.64)-(5.66)]_

- _In addition to have theoretical knowledge of the above, you should be able to
  perform all of it on practical examples – i.e., if you get a network with one
  hidden layer, a loss and a learning rate, you should perform the forward
  propagation, compute the loss, perform backpropagation and update weights
  using SGD. In order to do so, you should be able to derivate softmax with NLL,
  sigmoid with NLL and linear output with MSE._

- Stochastic gradient descent algorithm improvements _(you should be able to
  write the algorithms down and understand motivations behind them)_:
    - learning-rate decay
    - SGD with momentum _[Section 8.3.2 and Algorithm 8.2]_
    - SGD with Nestorov Momentum _(and how it is different from normal momentum)_ _[Section 8.3.3 and Algorithm 8.3]_
    - AdaGrad _(you should be able to explain why, in case of stationary
      gradient distribution, AdaGrad effectively decays learning rate)_
      _[Section 8.5.1 and Algorithm 8.4]_
    - RMSProp _(and why is it a generalization of AdaGrad)_ _[Section 8.5.2 and Algorithm 8.5]_
    - Adam _(and why the bias-correction terms (1-β^t) are there)_ _[Section 8.5.3 and Algorithm 8.7]_

- Regularization methods:
    - Early stopping _[Section 7.8, without the **How early stopping acts as a regularizer** part]_
    - L2 regularization _[First paragraph of 7.1.1 and Equation (7.5)]_
    - L1 regularization _[Section 7.1.2 up to Equation (7.20)]_
    - Dropout _[just the description of the algorithm]_
    - Batch normalization _[Section 8.7.1]_

- Gradient clipping _[Section 10.11.1]_

- Convolutional networks:
    - Basic convolution and cross-correlation operation on 4D tensors _[Equations (9.5) and (9.6)]_
    - Differences compared to a fully connected layer _[Section 9.2 and Figure 9.6]_
    - Multiple channels in a convolution _[Equation (9.7)]_
    - Stride and padding schemes _[Section 9.5 up to page 349, notably Equation (9.8)]_
    - Max pooling and average pooling _[Section 9.3]_
    - AlexNet _[general architecture, without knowing specific constants, i.e.,
      the following image which is taken from **Alex Krizhevsky et al.: ImageNet
      Classification with Deep Convolutional Neural Networks**
      https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf ]_

      ![AlexNet](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/exam/images/alexnet.svg)
    - ResNet _[only the important ideas and overall architecture of ResNet 151,
      without specific constants; the following is taken from **Kaiming He et
      al.: Deep Residual Learning for Image Recognition**
      https://arxiv.org/abs/1512.03385 ]_

      ![ResNet Block](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/exam/images/resnet-block.svg)
      ![ResNet Overview](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/exam/images/resnet-table.svg)
    - Object detection using Fast R-CNN _[overall architecture, RoI-pooling
      layer, parametrization of generated bounding boxes, used loss function;
      **Ross Girshick: Fast R-CNN** https://arxiv.org/abs/1504.08083 ]_
    - Proposing RoIs using Faster R-CNN _[overall architecture, the differences
      and similarities of Fast R-CNN and the proposal network from Faster R-CNN;
      **Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun: Faster R-CNN: Towards
      Real-Time Object Detection with Region Proposal Networks**
      https://arxiv.org/abs/1506.01497 ]_
    - Image segmentation with Mask R-CNN _[overall architecture, RoI-align layer;
      **Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick: Mask R-CNN** https://arxiv.org/abs/1703.06870 ]_

- Recurrent networks:
    - Using RNNs to represent sequences _[Figure 10.2 with `h` as output;
      Chapter 10 and Section 10.1]_
    - Using RNNs to classify every sequence element _[Figure 10.3; details in
      Section 10.2 excluding Sections 10.2.1-10.2.4]_
    - Bidirectional RNNs _[Section 10.3]_
    - Encoder-decoder sequence-to-sequence RNNs _[Section 10.4; note that you
      should know how the network is trained and also how it is later used to
      predict sequences]_
    - Stacked (or multi-layer) LSTM _[Figure 10.13a of Section 10.10.5; more
      details (not required for the exam) can be found in **Alex Graves:
      Generating Sequences With Recurrent Neural Networks**
      https://arxiv.org/abs/1308.0850 ]_
    - The problem of vanishing and exploding gradient _[Section 10.7]_
    - Long Shoft-Term Memory (LSTM) _[Section 10.10.1]_
    - Gated Recurrent Unit (GRU) _[Section 10.10.2]_

- Word representations _[in all cases, you should be able to describe the
  algorithm for computing the embedding, and how the backpropagation works
  (there is usually nothing special, but if I ask what happens if a word occurs
  multiple time in a sentence, you should be able to answer)]_
    - The `word2vec` word embeddings
        - CBOW and Skip-gram models _[**Tomas Mikolov, Kai Chen, Greg Corrado,
          Jeffrey Dean: Efficient Estimation of Word Representations in Vector
          Space** https://arxiv.org/abs/1301.3781 ]_
        - Hierarchical softmax _[Section 12.4.3.2, or Section 2.1 of the following paper]_
        - Negative sampling _[Section 2.2 of **Tomas Mikolov, Ilya Sutskever,
          Kai Chen, Greg Corrado, Jeffrey Dean: Distributed Representations of
          Words and Phrases and their Compositionality**
          https://arxiv.org/abs/1310.4546 ]_; note that negative sampling is
          a simplification of Importance sampling described in Section 12.4.3.3,
          with `w_i=1`; the proposal distribution in `word2vec` being unigram
          distribution to the power of 3/4
    - Character-level embeddings using RNNs _[C2W model from **Wang Ling, Tiago
      Luís, Luís Marujo, Ramón Fernandez Astudillo, Silvio Amir, Chris Dyer,
      Alan W. Black, Isabel Trancoso: Finding Function in Form: Compositional
      Character Models for Open Vocabulary Word Representation**
      https://arxiv.org/abs/1508.02096 ]_
    - Character-level embeddings using CNNs _[CharCNN from **Yoon Kim, Yacine
      Jernite, David Sontag, Alexander M. Rush: Character-Aware Neural Language
      Models** https://arxiv.org/abs/1508.06615 ]_

- Highway Networks _[**Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber:
  Training Very Deep Networks** https://arxiv.org/abs/1507.06228 ]_

- Machine Translation
    - Translation using encoder-decoder (also called sequence-to-sequence)
      architecture _[Sections 10.4 and Section 12.4.5]_
    - Attention mechanism in NMT _[Section 12.4.5.1, but you should also know
      the equations for the attention, notably Equations (4), (5), (6) and
      (A.1.2) of **Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio: Neural
      Machine Translation by Jointly Learning to Align and Translate**
      https://arxiv.org/abs/1409.0473 ]_
    - Subword units _[The BPE algorithm from Section 3.2 of **Rico Sennrich,
      Barry Haddow, Alexandra Birch: Neural Machine Translation of Rare Words
      with Subword Units** https://arxiv.org/abs/1508.07909 ]_

- Deep generative models using differentiable generator nets _[Section 20.10.2]_:
    - Variational autoencoders _[Section 20.10.3 up to page 698 (excluding),
      together with Reparametrization trick from Section 20.9 (excluding Section
      20.9.1)]_
        - Regular autoencoders _[undercomplete AE – Section 14.1, sparse AE
          – first two paragraphs of Section 14.2.1, denoising AE – Section
          14.2.2]_
    - Generative Adversarial Networks _[Section 20.10.4 up to page 702 (excluding)]_

- Structured Prediction
    - Conditional Random Fields (CRF) loss _[Sections 3.4.2 and A.7 of **R.
      Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, P. Kuksa:
      Natural Language Processing (Almost) from Scratch**
      http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf ]_
    - Connectionist Temporal Classification (CTC) loss *[**A. Graves, S.
      Fernández, F. Gomez, J. Schmidhuber: Connectionist Temporal
      Classification: Labelling Unsegmented Sequence Data with Recurrent Neural
      Networks** https://www.cs.toronto.edu/~graves/icml_2006.pdf ]*

- Reinforcement learning _[note that proofs are not required for reinforcement
  learning; all references are to the Mar 2018 draft of second edition of
  **Reinforcement Learning: An Introduction by Richar S. Sutton**
  http://incompleteideas.net/book/bookdraft2018mar21.pdf ]_
    - Multi-arm bandits _[Chapter 2, Sections 2.1-2.5]_
    - General setting of reinforcement learning _[agent-environment, action-state-reward, return; Chapter 3, Sections 3.1-3.3]_
    - Monte Carlo reinforcement learning algorithm _[Sections 5.1-5.4, especially the algorithm in Section 5.4]_
    - Temporal Difference RL Methods _[Section 6.1]_
    - SARSA algorithm _[Section 6.4]_
    - Q-Learning _[Section 6.5; you should also understand Eq. (6.1) and (6.2)]_
    - Policy gradient methods _[representing policy by the network, using
      softmax, Section 13.1]_
        - Policy gradient theorem _[Section 13.2]_
        - REINFORCE algorithm _[Section 13.3; note that the `γ^t` on the last
          line should not be there]_
        - REINFORCE with baseline algorithm _[Section 13.4; note that the `γ^t`
          on the last two lines should not be there]_
