# Deep Learning Labs 01

In order to run the examples, you need TensorFlow installed.

## Installing TensorFlow

Generally, you can follow the [official Download and Setup guide](https://www.tensorflow.org/versions/master/get_started/os_setup.html).

Extra simple version to install TensorFlow 0.11rf0 for Linux:
- Installing to your user Python packages:
  - Python 2.7: `pip install --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl`
  - Python 3.4: `pip install --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp34-cp34m-linux_x86_64.whl`
  - Python 3.5: `pip install --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp35-cp35m-linux_x86_64.whl`
- Installing to an virtual environment:
  - `virtualenv directory_for_the_new_virtual_environment`
  - `directory_for_the_new_virtual_environment/bin/pip install --user tensorflow_wheel`

Note that Windows version is not yet officially supported (as of 10 Oct 2016),
but it should be available very soon (see https://github.com/tensorflow/tensorflow/issues/17).
