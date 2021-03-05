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

### TOCEntry: ReCodEx

- _What are the tests used by ReCodEx_

  The tests used by ReCodEx correspond to the examples from the course website
  (unless stated otherwise), but they use a different random seed (so the
  results are not the same), and sometimes they use smaller number of
  epochs/iterations to finish sooner.

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
