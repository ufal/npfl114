### TOC: FAQ

### FAQEntry: tf.data

- _How to look what is in a `tf.data.Dataset`?_

  The `tf.data.Dataset` is not just an array, but a description of a pipeline,
  which can produce data if requested. A simple way to run the pipeline is
  to iterate it using Python iterators:
  ```python
  dataset = tf.data.Dataset.range(10)
  for entry in dataset:
      print(entry)
  ```

- _How to use `tf.data.Dataset` with `model.fit` or `model.evaluate`?_

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

- _Is every iteration through a `tf.data.Dataset` the same?_

  No. Because the dataset is only a _pipeline_ generating data, it is called
  each time the dataset is iterated – therefore, every `.shuffle` is called
  in every iteration.

- _How to generate different random numbers each epoch during `tf.data.Dataset.map`?_

  When a global random seed is set, methods like `tf.random.uniform` generate
  the same sequence of numbers on each iteration.

  The easiest method I found is to create a Generator object and use it to
  produce random numbers.

  ```python
  generator = tf.random.experimental.Generator.from_seed(42)
  data = tf.data.Dataset.from_tensor_slices(tf.zeros(10, tf.int32))
  data = data.map(lambda x: x + generator.uniform([], maxval=10, dtype=tf.int32))
  for _ in range(3):
      print(*[element.numpy() for element in data])
  ```

- _How to call numpy methods or other non-tf functions in `tf.data.Dataset.map`?_

  You can use [tf.numpy_function](https://www.tensorflow.org/api_docs/python/tf/numpy_function)
  to call a numpy function even in a computational graph. However, the results
  have no static shape information and you need to set it manually – ideally
  using [tf.ensure_shape](https://www.tensorflow.org/api_docs/python/tf/ensure_shape),
  which both sets the static shape and verifies during execution that the real
  shape mathes it.

  For example, to use the `bboxes_training` method from
  [bboxes_utils](#bboxes_utils), you could do something like:

  ```python
  anchors = np.array(...)

  def prepare_data(example):
      anchor_classes, anchor_bboxes = tf.numpy_function(
          bboxes_utils.bboxes_training, [anchors, example["classes"], example["bboxes"], 0.5], (tf.int32, tf.float32))
      anchor_classes = tf.ensure_shape(anchor_classes, [len(anchors)])
      anchor_bboxes = tf.ensure_shape(anchor_bboxes, [len(anchors), 4])
      ...
  ```

- _How to use `ImageDataGenerator` in `tf.data.Dataset.map`?_

  The `ImageDataGenerator` offers a `.random_transform` method, so we can use
  `tf.numpy_function` from the previous answer:

  ```python
  train_generator = tf.keras.preprocessing.image.ImageDataGenerator(...)

  def augment(image, label):
      return tf.ensure_shape(
          tf.numpy_function(train_generator.random_transform, [image], tf.float32),
          image.shape
      ), label

  dataset.map(augment)
  ```

### FAQEntry: Finetuning

- _How to make a part of the network frozen, so that its weights are not updated?_

  Each `tf.keras.layers.Layer`/`tf.keras.Model` has a mutable `trainable`
  property indicating whether its variables should be updated – however, after
  changing it, you need to call `.compile` again (or otherwise make sure the
  list of trainable variables for the optimizer is updated).

  Note that once `trainable == False`, the insides of a layer are no longer
  considered, even if some its sub-layers have `trainable == True`. Therefore, if
  you want to freeze only some sub-layers of a layer you use in your model, the
  layer itself must have `trainable == True`.

- _How to choose whether dropout/batch normalization is executed in training or
  inference regime?_

  When calling a `tf.keras.layers.Layer`/`tf.keras.Model`, a named option
  `training` can be specified, indicating whether training or inference regime
  should be used. For a model, this option is automatically passed to its layers
  which require it, and Keras automatically passes it during
  `model.{fit,evaluate,predict}`.

  However, you can manually pass for example `training=False` to a layer when
  using Functional API, meaning that layer is executed in the inference
  regime even when the whole model is training.

- _How does `trainable` and `training` interact?_

  The only layer, which is influenced by both these options, is batch
  normalization, for which:
  - if `trainable == False`, the layer is always executed in inference regime;
  - if `trainable == True`, the training/inference regime is chosen according
    to the `training` option.

### FAQEntry: Masking

- _How can sequences of different length be processed by a RNN?_

  Keras employs [masking](https://www.tensorflow.org/guide/keras/masking_and_padding)
  to indicate, which sequence elements are _valid_ and which are just _padding_.

  Usually, a mask is created using
  a [Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)
  or [Masking](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking) layer
  and is then automatically propagated. If `model.compile` is used, it is also
  automatically utilized in losses and metrics.

  However, in order for the mask propagation to work, you can use only
  `tf.keras.layers` to process the data, not raw TF operations like `tf.concat`
  or even the `+` operator (see `tf.keras.layers.Concatenate/Add/...`).

- _How to compute masked losses and masked metrics manually?_

  When you want to compute the losses and metrics manually, pass the mask as the
  third argument to their `__call__` method (each individual component of
  loss/metric is then multiplied by the mask, zeroing out the ones for padding
  elements).

- _How to print output masks of a `tf.keras.Model`?_

  When you call the model directly, like `model(input_batch)`, the mask of each
  output is available in a private `._keras_mask` property, so for single-output
  models you can print it with `print(model(input_batch)._keras_mask)`.

### FAQEntry: TensorBoard

- _How to create TensorBoard logs manually?_

  Start by creating a [SummaryWriter](https://www.tensorflow.org/api_docs/python/tf/summary/SummaryWriter)
  using for example:
  ```python
  writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
  ```
  and then you can generate logs inside a `with writer.as_default()` block.

  You can either specify `step` manually in each call, or you can use
  `tf.summary.experimental.set_step(step)`. Also, during training you
  usually want to log only some batches, so the logging block during
  training usually looks like:
  ```python
  if optimizer.iterations % 100 == 0:
      tf.summary.experimental.set_step(optimizer.iterations)
      with self._writer.as_default():
          # logging
  ```

- _What can be logged in TensorBoard?_
  - scalar values:
    ```python
    tf.summary.scalar(name like "train/loss", value, [step])
    ```
  - tensor values displayed as histograms or distributions:
    ```python
    tf.summary.histogram(name like "train/output_layer", tensor value castable to `tf.float64`, [step])
    ```
  - images as tensors with shape `[num_images, h, w, channels]`, where
    `channels` can be 1 (grayscale), 2 (grayscale + alpha), 3 (RGB), 4 (RGBA):
    ```python
    tf.summary.image(name like "train/samples", images, [step], [max_outputs=at most this many images])
    ```
  - possibly large amount of text (e.g., all hyperparameter values, sample
    translations in MT, …) in Markdown format:
    ```python
    tf.summary.text(name like "hyperparameters", markdown, [step])
    ```
  - audio as tensors with shape `[num_clips, samples, channels]` and values in $[-1,1]$ range:
    ```python
    tf.summary.audio(name like "train/samples", clips, sample_rate, [step], [max_outputs=at most this many clips])
    ```
