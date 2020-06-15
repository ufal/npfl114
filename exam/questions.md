####Questions@: ,Lecture 2 Questions

<div style="float: right">
#### Include: network.svg
</div>

- **Training Neural Network**  
  Assume the artificial neural network on the right, with mean square error loss
  and gold output of 3. Compute the values of all weights $w_i$ after performing
  an SGD update with learning rate 0.1.

  _Different networks architectures, activation functions (`tanh`, `sigmoid`,
  `softmax`) and losses (`MSE`, `NLL`) may appear in the exam._

- **Maximum Likelihood Estimation**  
  Formulate maximum likelihood estimator for neural network parameters and derive
  the following two losses:
  - NLL (negative log likelihood) loss for networks returning a probability distribution
  - MSE (mean square error) loss for networks returning a real number with
    a normal distribution with a fixed variance

- **Backpropagation Algorithm, SGD with Momentum**  
  Write down the backpropagation algorithm. Then, write down the SGD algorithm
  with momentum. Finally, formulate SGD with Nestorov momentum and explain the
  difference to SGD with regular momentum.

- **Adagrad and RMSProp**  
  Write down the AdaGrad algorithm and show that it tends to internally decay
  learning rate by a factor of $1/\sqrt{t}$ in step $t$. Furthermore, write
  down RMSProp algorithm and compare it to Adagrad.

- **Adam**  
  Write down the Adam algorithm and explain the bias-correction terms
  $(1-\beta^t)$.

####Questions@: ,Lecture 3 Questions

- **Regularization**  
  Define overfitting and sketch what a regularization is. Then describe
  basic regularization methods like early stopping, L2 and L1 regularization,
  dataset augmentation, ensembling and label smoothing.

- **Dropout**  
  Describe the dropout method and write down exactly how is it used during training and
  during inference. Then explain why it cannot be used on RNN state,
  describe the variational dropout variant, and also describe layer
  normalization.

- **Network Convergence**  
  Describe factors influencing network convergence, namely:
  - Parameter initialization strategies (explain also why batch normalization
    helps with the initialization range).
  - Problems with saturating non-linearities (and again, why batch normalization
    helps; also discuss why NLL (compared to MSE) helps with saturating non-linearities
    on the output layer).
  - Gradient clipping (and the difference between clipping individual gradient
    elements or the gradient as a whole).

####Questions@: ,Lecture 4 Questions

- **Convolution**  
  Write down equations of how convolution of a given image is computed. Assume the input
  is an image $I$ of size $H \times W$ with $C$ channels, the kernel $K$
  has size $N \times M$, the stride is $T \times S$, the operation performed is
  in fact cross-correlation (as usual in convolutional neural networks)
  and that $O$ output channels are computed. Explain both
  $\textit{SAME}$ and $\textit{VALID}$ padding schemes and write down output
  size of the operation for both these padding schemes.

- **Batch Normalization**  
  Describe the batch normalization method and explain how it is used during
  training and during inference. Explicitly write over what is being
  normalized in case of fully connected layers, and in case of convolutional
  layers. Compare batch normalization to layer normalization.

####Questions@: ,Lecture 5 Questions

- **VGG and ResNet**  
  Describe overall architecture of VGG and ResNet (you do not need to remember
  exact number of layers/filters, but you should know when a BatchNorm is
  executed, when ReLU, and how residual connections work when the number of
  channels increases). Then describe two ResNet extensions (WideNet, DenseNet,
  PyramidNet, ResNeXt).

- **CNN Regularization, SE, MBConv**  
  Describe CNN regularization methods (networks with stochastic depth, Cutout,
  DropBlock). Then show a Squeeze and excitation block for a ResNet
  and finally sketch mobile inverted bottleneck with separable convolutions.

- **Transposed Convolution**  
  Write down equations of how convolution of a given image is computed. Assume the input
  is an image $I$ of size $H \times W$ with $C$ channels, the kernel $K$
  has size $N \times M$, the stride is $S$, the operation performed is
  in fact cross-correlation (as usual in convolutional neural networks)
  and that $O$ output channels are computed. Then write down the equation of
  transposed convolution (or equivalently backpropagation through a convolution
  to its inputs).

####Questions@: ,Lecture 6 Questions

- **Two-stage Object Detection**  
  Define object detection task and describe Fast-RCNN and Faster-RCNN
  architectures. Notably, show what the overall architectures of the networks
  are, explain the RoI-pooling, show how the network parametrizes bounding
  boxes, how do the losses looks like, how are RoI chosen during training,
  how the objects are predicted, and what region proposal network does.

- **Image Segmentation**  
  Define object detection and image segmentation tasks, and sketch a Faster-RCNN
  and Mask-RCNN architectures. Notably, show what the overall architecture of
  the networks is, explain the RoI-pooling and RoI-align layer, show how the network
  parametrizes bounding boxes, how do the losses looks like, how are RoI chosen
  during training and how the objects are predicted.

- **Single-stage Object Detection**  
  Define object detection task and describe single-stage detector architecture.
  Namely, show feature pyramid network, define focal loss and sketch RetinaNet
  – the overall architecture including the convolutional classification and
  bounding box prediction heads, overall loss, how the gold labels are
  generated, and how the objects are predicted.

####Questions@: ,Lecture 7 Questions

- **LSTM**  
  Write down how the Long Short-Term Memory cell operates.

- **GRU and Highway Networks**  
  Show a basic RNN cell (using just one hidden layer) and then write down
  how it is extended using gating into the Gated Recurrent Unit.
  Finally, describe highway networks and compare them to RNN.

####Questions@: ,Lecture 8 Questions

- **Character-level word embeddings**  
  Describe why are character-level word embeddings useful. Then describe the
  two following methods:
  - RNN: using bidirectional recurrent neural networks
  - CNN: describe how convolutional networks (CNNs) can be used to compute
    character-level word embeddings.  Write down the exact equation computing
    the embedding, assuming that the input word consists of characters
    $\{x_1, \ldots, x_N\}$ represented by embeddings $\{e_1, \ldots, e_N\}$ for
    $e_i \in \mathbb R^D$, and we use $F$ filters of widths $w_1, \ldots, w_F$.
    Also explicitly count the number of parameters.

- **Sequence classification and CRF**  
  Describe how RNNs, bidirectional RNNs and multi-layer RNNs can be used to
  classify every element of a given sequence (i.e., what the architecture of
  a tagger might be; include also residual connections and suitable places
  for dropout layers). Then, explain how a CRF layer works, define score
  computation for a given sequence of inputs and sequence of labels,
  describe the loss computation during training, and sketch the inference
  algorithm.

- **CTC Loss**  
  Describe CTC loss and the whole settings which can be solved utilizing CTC
  loss. Then show how CTC loss can be computed. Finally, describe greedy
  and beam search CTC decoding.

####Questions@: ,Lecture 9 Questions

- **Word2vec and Hierarchical and Negative Sampling**  
  Explain how can word embeddings be precomputed using the CBOW and Skip-gram
  models. First start with the variants where full softmax is performed, and
  then describe how hierarchical softmax and negative sampling is used to speedup
  training of word embeddings.

- **Neural Machine Translation and Attention**  
  Draw/write how an encoder-decoder architecture is used for machine translation,
  both during training and during inference. Then describe the architecture
  of an attention module.

- **Neural Machine Translation and Subwords**  
  Draw/write how an encoder-decoder architecture is used for machine translation,
  both during training and during inference (without attention). Furthermore,
  elaborate on how subword units are used to reduce out-of-vocabulary problem and
  sketch BPE algorithm and WordPieces algorithm for constructing fixed number of
  subword units.

####Questions@: ,Lecture 10 Questions

- **Variational Autoencoders**  
  Describe deep generative modelling using variational autoencoders – show VAE
  architecture, devise training algorithm, write training loss, and propose sampling
  procedure.

- **Generative Adversarial Networks**  
  Describe deep generative modelling using generative adversarial networks -- show GAN
  architecture and describe training procedure and training loss. Mention also
  CGAN (conditional GAN) and sketch generator and discriminator architecture in a DCGAN.

####Questions@: ,Lecture 11 Questions

- **Reinforcement learning**  
  Describe the general reinforcement learning settings and formulate the Monte
  Carlo algorithm. Then, formulate and prove the policy gradient theorem
  and write down the REINFORCE algorithm.

- **REINFORCE with baseline**  
  Describe the general reinforcement learning settings, formulate the
  policy gradient theorem and write down the REINFORCE algorithm.
  Then explain what is the baseline, show policy gradient theorem with the
  baseline (including the proof of why the baseline can be included),
  and write down the REINFORCE with baseline algorithm.

####Questions@: ,Lecture 12 Questions

- **Speech Synthesis**  
  Describe the WaveNet network (what a dilated convolution and gated activations
  are, how the residual block looks like, what the overall architecture is, and how
  global and local conditioning work). Discuss parallelizability of training and
  inference, show how Parallel WaveNet can speedup inference, and sketch how it is
  trained.

- **Neural Turing Machines**  
  Sketch an overall architecture of a Neural Turing Machine with an LSTM
  controller, assuming $R$ reading heads and one write head. Describe the
  addressing mechanism (content addressing and its combination with previous
  weights, shifts, and sharpening) and reading and writing operations. Finally,
  describe the inputs and the outputs of the controller.

####Questions@: ,Lecture 13 Questions

- **Transformer**  
  Describe Transformer architecture, namely the self-attention layer, multi-head
  self-attention layer, masked self-attention and overall architecture of an
  encoder and a decoder. Describe positional embeddings, learning rate schedule
  during training and parallelizability of training and inference.

- **BERT**  
  Describe the BERT model architecture (including multi-head self-attention layer)
  and its pre-training – format of input and output data, masked language model
  and next sentence prediction. Define GELU and describe how the BERT model
  can be finetuned to perform POS tagging, sentiment analysis and paraphrase
  detection (detect if two sentences have the same meaning).
