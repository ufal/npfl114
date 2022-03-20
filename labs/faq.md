### TOC: FAQ

### TOCEntry: Install

- _Installing to central user packages repository_

  You can install all required packages to central user packages repository using
  `pip3 install --user tensorflow==2.8.0 tensorflow-addons==0.16.1 tensorflow-probability==0.16.0 tensorflow-hub==0.12.0 gym==0.20.0 scipy`.

- _Installing to a virtual environment_

  Python supports virtual environments, which are directories containing
  independent sets of installed packages. You can create a virtual environment
  by running `python3 -m venv VENV_DIR` followed by
  `VENV_DIR/bin/pip3 install tensorflow==2.8.0 tensorflow-addons==0.16.1 tensorflow-probability==0.16.0 tensorflow-hub==0.12.0 gym==0.20.0 scipy`
  (or `VENV_DIR/Scripts/pip3` on Windows).

- _**Windows** installation_

  - On Windows, it can happen that `python3` is not in PATH, while `py` command
    is – in that case you can use `py -m venv VENV_DIR`, which uses the newest
    Python available, or for example `py -3.9 -m venv VENV_DIR`, which uses
    Python version 3.9.

  - If your Windows TensorFlow fails with `ImportError: DLL load failed`,
    you are probably missing
    [Visual C++ 2019 Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

  - If you encounter a problem creating the logs in the `args.logdir` directory,
    a possible cause is that the path is longer than 260 characters, which is
    the default maximum length of a complete path on Windows. However, you can
    increase this limit on Windows 10, version 1607 or later, by following
    the [instructions](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).

- _**macOS** installation_

  - With an **Intel** processor, you should not need anything special.

  - If you have **Apple Silicon**, the installation is a bit more involved,
    because some Python packages do not yet have an official Arm64 binary build.
    The easiest workaround is to use Conda, which contains all the required
    dependencies.

    - Download [Conda env](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh).

    - Install it and activate it.
      ```
      chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
      sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
      source ~/miniforge3/bin/activate
      ```

    - Install the Arm64 TensorFlow dependencies.
      ```
      conda install -c apple tensorflow-deps==2.8.0
      ```

    - Install the Arm64 build of TensorFlow.
      ```
      python -m pip install tensorflow-macos==2.8.0
      ```

- _**GPU** support on Linux and Windows_

  TensorFlow 2.8 supports NVIDIA GPU out of the box, but you need to install
  CUDA 11.2 and cuDNN 8.1 libraries yourself.

- _**GPU** support on macOS_

  The AMD and Apple Silicon GPUs can be used by installing
  a plugin providing the GPU acceleration using:
  ```
  python -m pip install tensorflow-metal
  ```

- _Errors when running with a GPU_

  If you encounter errors when running with a GPU:
  - if you are using the GPU also for displaying, try using the following
    environment variable: `export TF_FORCE_GPU_ALLOW_GROWTH=true`
  - you can rerun with `export TF_CPP_MIN_LOG_LEVEL=0` environmental variable,
    which increases verbosity of the log messages.

### TOCEntry: MetaCentrum

- _How to install TensorFlow dependencies on MetaCentrum?_

  To install CUDA, cuDNN and Python 3.8 on MetaCentrum, it is enough to run
  in every session the following command:
  ```
  module add python/3.8.0-gcc-rab6t cuda/cuda-11.2.0-intel-19.0.4-tn4edsz cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t
  ```

- _How to install TensorFlow on MetaCentrum?_

  Once you have the required dependencies, you can create a virtual environment
  and install TensorFlow in it. However, note that by default the MetaCentrum
  jobs have a little disk space, so read about
  [how to ask for scratch storage](https://wiki.metacentrum.cz/wiki/Scratch_storage)
  when submitting a job, and about [quotas](https://wiki.metacentrum.cz/wiki/Quotas),

  TL;DR:
  - Run an interactive CPU job, asking for 16GB scratch space:
    ```
    qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb -I
    ```

  - In the job, use the allocated scratch space as a temporary directory:
    ```
    export TMPDIR=$SCRATCHDIR
    ```

  - Finally, create the virtual environment and install TensorFlow in it:
    ```
    module add python/3.8.0-gcc-rab6t cuda/cuda-11.2.0-intel-19.0.4-tn4edsz cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t
    python3 -m venv CHOSEN_VENV_DIR
    CHOSEN_VENV_DIR/bin/pip install --no-cache-dir --upgrade pip setuptools
    CHOSEN_VENV_DIR/bin/pip install --no-cache-dir tensorflow==2.8.0 tensorflow-addons==0.16.1 tensorflow-probability==0.16.0 tensorflow-hub==0.12.0 gym==0.20.0 scipy
    ```

- _How to run a GPU computation on MetaCentrum?_

  First, read the official MetaCentrum documentation:
  [Beginners guide](https://wiki.metacentrum.cz/wiki/Beginners_guide),
  [About scheduling system](https://wiki.metacentrum.cz/wiki/About_scheduling_system),
  [GPU clusters](https://wiki.metacentrum.cz/wiki/GPU_clusters).

  TL;DR: To run an interactive GPU job with 1 CPU, 1 GPU, 16GB RAM, and 8GB scatch
  space, run:
  ```
  qsub -q gpu -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_local=8gb -I
  ```

  To run a script in a non-interactive way, replace the `-I` option with the script to be executed.

  If you want to run a CPU-only computation, remove the `-q gpu` and `ngpus=1:`
  from the above commands.

### TOCEntry: AIC

- _How to install TensorFlow dependencies on [AIC](https://aic.ufal.mff.cuni.cz)?_

  To install CUDA, cuDNN and Python 3.9 on AIC, you should add the following to
  your `.profile`:
  ```
  export PATH="/lnet/aic/data/python/3.9.9/bin:$PATH"
  export LD_LIBRARY_PATH="/lnet/aic/opt/cuda/cuda-11.2/lib64:/lnet/aic/opt/cuda/cuda-11.2/cudnn/8.1.1/lib64:/lnet/aic/opt/cuda/cuda-11.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
  ```

- _How to run a GPU computation on AIC?_

  First, read the official AIC documentation:
  [Submitting CPU Jobs](https://aic.ufal.mff.cuni.cz/index.php/Submitting_CPU_Jobs),
  [Submitting GPU Jobs](https://aic.ufal.mff.cuni.cz/index.php/Submitting_GPU_Jobs).

  TL;DR: To run an interactive GPU job with 1 CPU, 1 GPU, and 16GB RAM, run:
  ```
  qrsh -q gpu.q -l gpu=1,mem_free=16G,h_data=16G -pty yes bash -l
  ```

  To run a script requiring a GPU in a non-interactive way, use
  ```
  qsub -q gpu.q -l gpu=1,mem_free=16G,h_data=16G -cwd -b y SCRIPT_PATH
  ```

  If you want to run a CPU-only computation, remove the `-q gpu.q` and `gpu=1,`
  from the above commands.

### TOCEntry: Git

- _Is it possible to keep the solutions in a Git repository?_

  Definitely. Keeping the solutions in a branch of your repository,
  where you merge them with the course repository, is probably a good idea.
  However, please keep the cloned repository with your solutions **private**.

- _On GitHub, do not create a **public** fork with your solutions_

  If you keep your solutions in a GitHub repository, please do not create
  a clone of the repository by using the Fork button – this way, the cloned
  repository would be **public**.

  Of course, if you just want to create a pull request, GitHub requires a public
  fork and that is fine – just do not store your solutions in it.

- _How to clone the course repository?_

  To clone the course repository, run
  ```
  git clone https://github.com/ufal/npfl114
  ```
  This creates the repository in the `npfl114` subdirectory; if you want a different
  name, add it as a last parameter.

  To update the repository, run `git pull` inside the repository directory.

- _How to keep the course repository as a branch in your repository?_

  If you want to store the course repository just in a local branch of your
  existing repository, you can run the following command while in it:
  ```
  git remote add upstream https://github.com/ufal/npfl114
  git fetch upstream
  git checkout -t upstream/master
  ```
  This creates a branch `master`; if you want a different name, add
  `-b BRANCH_NAME` to the last command.

  In both cases, you can update your checkout by running `git pull` while in it.

- _How to merge the course repository with your modifications?_

  If you want to store your solutions in a branch merged with the course
  repository, you should start by
  ```
  git remote add upstream https://github.com/ufal/npfl114
  git pull upstream master
  ```
  which creates a branch `master`; if you want a different name,
  change the last argument to `master:BRANCH_NAME`.

  You can then commit to this branch and push it to your repository.

  To merge the current course repository with your branch, run
  ```
  git merge upstream master
  ```
  while in your branch. Of course, it might be necessary to resolve conflicts
  if both you and I modified the same place in the templates.

### TOCEntry: ReCodEx

- _What files can be submitted to ReCodEx?_

  You can submit multiple files of any type to ReCodEx. There is a limit of
  **20** files per submission, with a total size of **20MB**.

- _What file does ReCodEx execute and what arguments does it use?_

  Exactly one file with `py` suffix must contain a line starting with `def main(`.
  Such a file is imported by ReCodEx and the `main` method is executed
  (during the import, `__name__ == "__recodex__"`).

  The file must also export an argument parser called `parser`. ReCodEx uses its
  arguments and default values, but it overwrites some of the arguments
  depending on the test being executed – the template should always indicate which
  arguments are set by ReCodEx and which are left intact.

- _What are the time and memory limits?_

  The memory limit during evaluation is **1.5GB**. The time limit varies, but it should
  be at least 10 seconds and at least twice the running time of my solution.

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

  When a GPU is visible, you should create the `generator` explicitly on a CPU
  using a `with tf.device("/cpu:0"):` block (on macOS, it will crash otherwise).

- _How to call numpy methods or other non-tf functions in `tf.data.Dataset.map`?_

  You can use [tf.numpy_function](https://www.tensorflow.org/api_docs/python/tf/numpy_function)
  to call a numpy function even in a computational graph. However, the results
  have no static shape information and you need to set it manually – ideally
  using [tf.ensure_shape](https://www.tensorflow.org/api_docs/python/tf/ensure_shape),
  which both sets the static shape and verifies during execution that the real
  shape matches it.

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

- _How to use linear warmup?_

  You can prepend any `following_schedule` by using the following `LinearWarmup`
  schedule:

  ```python
  class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
      def __init__(self, warmup_steps, following_schedule):
          self._warmup_steps = warmup_steps
          self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
          self._following = following_schedule

      def __call__(self, step):
          return tf.cond(step < self._warmup_steps,
                         lambda: self._warmup(step),
                         lambda: self._following(step - self._warmup_steps))
  ```

### TOCEntry: TensorBoard

- _Cannot start TensorBoard after installation_

  If `tensorboard` executable cannot be found, make sure the directory with pip installed
  packages is in your PATH (that directory is either in your virtual environment
  if you use a virtual environment, or it should be `~/.local/bin` on Linux
  and `%UserProfile%\AppData\Roaming\Python\Python3[7-9]` and
  `%UserProfile%\AppData\Roaming\Python\Python3[7-9]\Scripts` on Windows).

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
