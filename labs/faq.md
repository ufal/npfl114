### TOC: FAQ

### TOCEntry: Install

- _Installing to central user packages repository_

  You can install all required packages to central user packages repository using
  `pip3 install --user --upgrade pip setuptools` followed by
  `pip3 install --user tensorflow==2.4.1 tensorflow-addons==0.12.1 tensorflow-probability==0.12.1 tensorflow-hub==0.11.0 gym==0.18.0`.

- _Installing to a virtual environment_

  Python supports virtual environments, which are directories containing
  independent sets of installed packages. You can create a virtual environment
  by running `python3 -m venv VENV_DIR` and then install the required packages with
  `VENV_DIR/bin/pip3 install --upgrade pip setuptools` followed by
  `VENV_DIR/bin/pip3 install tensorflow==2.4.1 tensorflow-addons==0.12.1 tensorflow-probability==0.12.1 tensorflow-hub==0.11.0 gym==0.18.0`.

- _Installing to MetaCentrum_

  As of Apr 2021, the minimum CUDA version across MetaCentrum is 10.2, and the
  highest officially available CUDA+cuDNN is also 10.2. Therefore, I have build
  TensorFlow 2.4.1 for CUDA 10.2 and cuDNN 7.6 to use on MetaCentrum.

  During installation, start by using official Python 3.6 and CUDA+cuDNN
  packages via `module add python-3.6.2-gcc cuda/cuda-10.2.89-gcc-6.3.0-34gtciz
  cudnn/cudnn-7.6.5.32-10.2-linux-x64-gcc-6.3.0-xqx4s5f`. Note that this command
  must be always executed before using the installed TensorFlow.

  Then create a virtual environment by `python3 -m venv VENV_DIR` and
  install the required packages with `VENV_DIR/bin/pip3 install --upgrade pip setuptools` followed by
  `VENV_DIR/bin/pip3 install
  https://ufal.mff.cuni.cz/~straka/packages/tf/2.4/metacentrum/tensorflow-2.4.1-cp36-cp36m-linux_x86_64.whl
  https://ufal.mff.cuni.cz/~straka/packages/tf/2.4/metacentrum/tensorflow_addons-0.12.1-cp36-cp36m-linux_x86_64.whl
  tensorflow-probability==0.12.1 tensorflow-hub==0.11.0 gym==0.18.0`.

- _Windows TensorFlow fails with ImportError: DLL load failed_

  If your Windows TensorFlow fails with `ImportError: DLL load failed`,
  you are probably missing
  [Visual C++ 2019 Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

- _Cannot start TensorBoard after installation_

  If `tensorboard` cannot be found, make sure the directory with pip installed
  packages is in your PATH (that directory is either in your virtual environment
  if you use a virtual environment, or it should be `~/.local/bin` on Linux
  and `%UserProfile%\AppData\Roaming\Python\Python3[5-7]` and
  `%UserProfile%\AppData\Roaming\Python\Python3[5-7]\Scripts` on Windows).

### TOCEntry: Git

- _Is it possible to keep the solutions in a Git repository_

  Definitely, keeping the solutions in a branch of your repository,
  where you merge it with the course repository, is probably a good idea.
  However, please keep the cloned repository with your solutions **private**.

- _Do not create a **public** fork of the repository on Github_

  On Github, please do not create a clone of the repository by using the Fork
  button – this way, the cloned repository would be **public**.

- _How to clone the course repository_

  To clone the course repository, run
  ```
  git clone https://github.com/ufal/npfl114
  ```
  This creates the repository in `npfl114` subdirectory; if you want a different
  name, add it as a last parameter.

  If you want to store the repository just in a local branch of your existing
  repository, you can run the following command while in it:
  ```
  git remote add upstream https://github.com/ufal/npfl114
  git fetch upstream
  git checkout -t upstream/master
  ```
  This creates a branch `master`; if you want a different name, add
  `-b BRANCH_NAME` to the last command.

  In both cases, you can update your checkout by running `git pull` while in it.

- _How to merge the course repository with your modifications_

  If you want to store your solutions in a branch merged with the course
  repository, you should start by
  ```
  git remote add upstream https://github.com/ufal/npfl114
  git pull upstream master
  ```
  which creates a branch `master`; if you want a different name,
  change the last argument to `master:BRANCH_NAME`.

  You can then commit to this branch and push it to some central repository.

  To merge the current course repository with your branch, run
  ```
  git merge ustream master
  ```
  while in your branch. Of course, it might be necessary to resolve conflicts
  if both you and I modified the same place in the templates.

### TOCEntry: ReCodEx

- _What are the tests used by ReCodEx_

  The tests used by ReCodEx correspond to the examples from the course website
  (unless stated otherwise), but they use a different random seed (so the
  results are not the same), and sometimes they use smaller number of
  epochs/iterations to finish sooner.

### TOCEntry: Debugging

- _How to debug problems “inside” computation graphs with weird stack traces?_

  At the beginning of your program, run
  ```python
  tf.config.run_functions_eagerly(True)
  ```
  The `tf.funcion`s (with the exception of the ones used in `tf.data` pipelines)
  are then not traced (i.e., no computation graphs are created) and the pure
  Python code is executed instead.

- _How to debug problems “inside” `tf.data` pipelines with weird stack traces?_

  Unfortunately, the solution above does not affect tracing in `tf.data`
  pipelines (for example in `tf.data.Dataset.map`). However, since TF 2.5, the
  command
  ```python
  tf.data.experimental.enable_debug_mode()
  ```
  should disable any asynchrony, parallelism, or non-determinism and forces
  Python execution (as opposed to trace-compiled graph execution) of
  user-defined functions passed into transformations such as `tf.data.Dataset.map`.

### TOCEntry: GPU

- _Requirements for using a GPU_

  To use an NVIDIA GPU with TensorFlow 2.4, you need to install CUDA 11.0 and
  cuDNN 8.0 – see [the details about GPU support](https://www.tensorflow.org/install/gpu).

- _Errors when running with a GPU_

  If you encounter errors when running with a GPU:
  - if you are using the GPU also for displaying, try using the following
    environment variable: `export TF_FORCE_GPU_ALLOW_GROWTH=true`
  - you can rerun with `export TF_CPP_MIN_LOG_LEVEL=0` environmental variable,
    which increases verbosity of the log messages.

### TOCEntry: tf.ragged

- _Bug when RaggedTensors are used in backward/bidirectional direction and
  whole sequence is returned_

  In TF 2.4, **RaggedTensors processed by backward (and therefore also by
  bidirectional) RNNs produce bad results when whole sequences are returned**.
  (Producing only the last output or processing in forward direction is fine.)
  The problem has been [fixed in the master branch](https://github.com/tensorflow/tensorflow/commit/da96383680c0a320c9551e020c26132ae5ebb024)
  and [also in the TF 2.5 branch](https://github.com/tensorflow/tensorflow/pull/48887).

  A workaround is to use the manual to/from dense tensor conversion described
  in the next point.

- _Slow RNNs when using RaggedTensors on GPU_

  Unfortunately, the current LSTM/GRU implementation
  [does not use cuDNN acceleration when processing RaggedTensors](https://github.com/tensorflow/tensorflow/issues/48838).
  However, you can get around it by manually converting the RaggedTensors to
  dense before/after the layer, so when `inputs` is a `tf.RaggedTensor`,
  - if `rnn` is a `tf.keras.layers.LSTM/GRU/RNN/Bidirectional` layer producing
    a **single output**, you can use the following workaround:
    ```python
    outputs = rnn(inputs.to_tensor(), mask=tf.sequence_mask(inputs.row_lengths()))
    ```
  - if `rnn` is a `tf.keras.layers.LSTM/GRU/RNN/Bidirectional` layer producing
    a **whole sequence**, in addition to the above line you also need to convert
    the dense result back to a RaggedTensor via for example:
    ```python
    outputs = tf.RaggedTensor.from_tensor(outputs, inputs.row_lengths())
    ```

### TOCEntry: tf.data

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
  train = cags.train.map(lambda example: (example["image"], example["label"]))
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

  Instead, create a `Generator` object and use it to produce random numbers.

  ```python
  generator = tf.random.Generator.from_seed(42)
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
  [bboxes_utils](#bboxes_utils), you could proceed as follows:

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

### TOCEntry: Finetuning

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


### TOCEntry: TensorBoard

- _How to create TensorBoard logs manually?_

  Start by creating a [SummaryWriter](https://www.tensorflow.org/api_docs/python/tf/summary/SummaryWriter)
  using for example:
  ```python
  writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
  ```
  and then you can generate logs inside a `with writer.as_default()` block.

  You can either specify `step` manually in each call, or you can set
  it as the first argument of `as_default()`. Also, during training you
  usually want to log only some batches, so the logging block during
  training usually looks like:
  ```python
  if optimizer.iterations % 100 == 0:
      with self._writer.as_default(step=optimizer.iterations):
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
