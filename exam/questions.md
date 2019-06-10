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
    helps with this issue).
  - Problems with saturating non-linearities (and again, why batch normalization
    helps; you can also discuss why NLL helps with saturating non-linearities
    on the output layer).
  - Gradient clipping (and the difference between clipping individual weights
    or the gradient as a whole).

- **Convolution**  
  Write down equations of how convolution of a given image is computed. Assume the input
  is an image $I$ of size $H \times W$ with $C$ channels, the kernel $K$
  has size $N \times M$, the stride is $T \times S$, the operation performed is
  n fact cross-correlation (as usual in convolutional neural networks)
  and that $O$ output channels are computed. Explain both
  $\textit{SAME}$ and $\textit{VALID}$ padding schemes and write down output
  size of the operation for both these padding schemes.

- **Batch Normalization**  
  Describe the batch normalization method and explain how it is used during
  training and during inference. Explicitly write over what is being
  normalized in case of fully connected layers, and in case of convolutional
  layers. Compare batch normalization to layer normalization.

- **VGG and ResNet**  
  Describe overall architecture of VGG and ResNet (you do not need to remember
  exact number of layers/filters, but you should know when a BatchNorm is
  executed, when ReLU, and how residual connections work when the number of
  channels increases). Then describe two ResNet extensions (WideNet, DenseNet,
  PyramidNet, ResNeXt).

- **Object Detection and Segmentation**  
  Describe object detection and image segmentation tasks, and sketch Fast-RCNN, Faster-RCNN and
  Mask-RCNN architectures. Notably, show what the overall architectures of the networks
  are, explain the RoI-pooling and RoI-align layers, show how the network predicts RoI
  sizes, how do the losses looks like, how are RoI chosen during training and
  prediction, and what region proposal network does.

- **Object Detection**  
  Describe object detection task, and sketch Fast-RCNN, Faster-RCNN and
  RetinaNet architectures. Notably, show the overall architectures of the
  networks, explain the RoI-pooling layer, show how the network predicts RoI
  sizes, how do the losses looks like (classification loss, boundary prediction
  loss, focal loss for RetinaNet), and what a feature pyramid network is.

- **LSTM**  
  Write down how the Long Short-Term Memory cell operates.

- **GRU and Highway Networks**  
  Show a basic RNN cell (using just one hidden layer) and then write down
  how it is extended using gating into the Gated Recurrent Unit.
  Finally, describe highway networks and compare them to RNN.

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

- **Word2vec and Hierarchical and Negative Sampling**  
  Explain how can word embeddings be precomputed using the CBOW and Skip-gram
  models. First start with the variants where full softmax is performed, and
  then describe how hierarchical softmax and negative sampling is used to speedup
  training of word embeddings.

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

- **Neural Machine Translation and BPE**  
  Draw/write how an encoder-decoder architecture is used for machine translation,
  both during training and during inference, including attention. Furthermore,
  elaborate on how subword units are used to reduce out-of-vocabulary problem and
  sketch BPE algorithm for constructing fixed number of subword units.

- **Variational Autoencoders**  
  Describe deep generative modelling using variational autoencoders – show VAE
  architecture, devise training algorithm, write training loss, and propose sampling
  procedure.

- **Generative Adversarial Networks**  
  Describe deep generative modelling using generative adversarial networks -- show GAN
  architecture and describe training procedure and training loss. Mention also
  CGAN (conditional GAN) and sketch generator and discriminator architecture in a DCGAN.

- **Speech Synthesis**  
  Describe the WaveNet network (what a dilated convolution and gated activations
  are, how the residual block like, what the overall architecture is, and how
  global and local conditioning work). Discuss parallelizability of training and
  inference, show how Parallel WaveNet can speedup inference, and sketch how it is
  trained.

- **Reinforcement learning**  
  Describe the general reinforcement learning settings and describe the Monte
  Carlo algorithm. Then, formulate the policy gradient theorem (proof not
  needed), write down the REINFORCE algorithm, the REINFORCE with baseline
  algorithm, and sketch now it can be used to design the NasNet.

- **Transformer**  
  Describe Transformer architecture, namely the self-attention layer, multi-head
  self-attention layer, and overall architecture of an encoder and a decoder.
  Also discuss the positional embeddings.

- **Neural Turing Machines**  
  Sketch an overall architecture of a Neural Turing Machine with an LSTM
  controller, assuming $R$ reading heads and one write head. Describe the
  addressing mechanism (content addressing and its combination with previous
  weights, shifts, and sharpening), and reading and writing operations.
