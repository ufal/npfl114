### Assignment: explore_examples

Your goal in this zero-point assignment is to explore the prepared examples.
- The [example_keras_models.py](https://github.com/ufal/npfl114/tree/master/labs/03/example_keras_models.py)
  example demonstrates three different ways of constructing Keras models
  – sequential models, functional API and model subclassing.
- The [example_keras_manual_batches.py](https://github.com/ufal/npfl114/tree/master/labs/03/example_keras_manual_batches.py)
  shows how to train and evaluate Keras model when using custom batches.
- The [example_manual.py](https://github.com/ufal/npfl114/tree/master/labs/03/example_manual.py)
  illustrates how to implement a manual training loop without using
  `Model.compile`, with custom `Optimizer`, loss function and metric.
  However, this example is 2-3 times slower than the previous two ones.
- The [example_manual_tf_function.py](https://github.com/ufal/npfl114/tree/master/labs/03/example_manual_tf_function.py)
  uses `tf.function` annotation to speed up execution of the previous
  example back to the level of `Model.fit`. See the
  [official `tf.function` documentation](https://www.tensorflow.org/api_docs/python/tf/function)
  for details.
