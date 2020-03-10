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
    helps with this issue).
  - Problems with saturating non-linearities (and again, why batch normalization
    helps; you can also discuss why NLL helps with saturating non-linearities
    on the output layer).
  - Gradient clipping (and the difference between clipping individual gradient
    elements or the gradient as a whole).
