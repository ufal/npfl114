# Oct 24

- Softmax with NLL (negative log likelyhood) as a loss functioin [Section 6.2.2.3 of DLB, notably equation (6.30); you should also be able to compute derivative of softmax + NLL with respect to the inputs of the softmax]
- Gradient optimization algorithms (see [NAIL002 lecture](https://is.cuni.cz/studium/eng/predmety/index.php?do=predmet&kod=NAIL002) for detailed treatment of this topic)
  - SGD algorithm [Section 8.3.1 and Algorithm 8.1 of DLB]
  - Learning rate decay [`tf.train.exponential_decay`]
  - SGD with Momentum algorithm [Section 8.3.2 and Algorithm 8.2 of DLB]
  - SGD with Nestorov Momentum algorithm [Section 8.3.3 and Algorithm 8.3 of DLB]
- Optimization algorithms with adaptive gradients
  - AdaGrad algorithm [Section 8.5.1 and Algorithm 8.4 of DLB]
  - RMSProp algorithm [Section 8.5.2 and Algorithm 8.5 of DLB]
  - Adam algorithm [Section 8.5.3 and Algorithm 8.7 of DLB]
- Parameter initialization strategies [Section 8.4 of DLB]
