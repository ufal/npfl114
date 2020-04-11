### `tf.data.Dataset`

- How to look what is in a `tf.data.Dataset`?

  The `tf.data.Dataset` is not just an array, but a description of a pipeline,
  which can produce data if requested. A simple way to run the pipeline is
  to iterate it using Python iterators:
  ```python
  dataset = tf.data.Dataset.range(10)
  for entry in dataset:
      print(entry)
  ```

- How to use `tf.data.Dataset` with `model.fit` or `model.evaluate`?

  To use a `tf.data.Dataset` in Keras, the dataset elements should be pairs
  `(input_data, gold_labels)`, where `input_data` and `gold_labels` must be
  already batched. For example, given `CAGS` dataset, you can preprocess
  training data for `cags_classification` as (for development data, you would
  remove the `.shuffle`):
  ```python
  train = cags.train.map(CAGS.parse)
  train = train.map(lambda example: (example["image"], example["label"]))
  train = train.shuffle(10000, seed=args.seed)
  train = train.batch(args.batch_size)
  ```

- Is every iteration through a `tf.data.Dataset` the same?

  No. Because the dataset is only a _pipeline_ generating data, it is called
  each time the dataset is iterated – therefore, every `.shuffle` is called
  in every iteration.

  Similarly, if you use random numbers in a `augment` method and use it in
  a `.map(augment)`, it is called on each iteration and can modify the same image
  differently in different epochs.

  ```python
  data = tf.data.Dataset.from_tensor_slices(tf.zeros(10, tf.int32))
  data = data.map(lambda x: x + tf.random.uniform([], maxval=10, dtype=tf.int32))
  for _ in range(3):
      print(*[element.numpy() for element in data])
  ```
