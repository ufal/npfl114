The tasks are evaluated automatically using the [ReCodEx Code
Examiner](https://recodex.mff.cuni.cz/). The evaluation is
performed using Python 3.6, TensorFlow 2.1.0, TensorFlow Addons 0.8.1,
TensorFlow Hub 0.7.0, TensorFlow Probability 0.9.0, OpenAI Gym 0.15.4
and NumPy 1.18.1.

#### Installing to Central User Packages Repository

You can install all required packages to central user packages repository using
`pip3 install --user --upgrade pip setuptools` followed by
`pip3 install --user tensorflow==2.1.0 tensorflow-addons==0.8.1
tensorflow-hub==0.7.0 tensorflow-probability==0.9.0 gym==0.15.4`.

#### Installing to a Virtual Environment

Python supports virtual environments, which are directories containing
independent sets of installed packages. You can create the virtual environment
by running `python3 -m venv VENV_DIR` followed by
`VENV_DIR/bin/pip3 install --upgrade pip setuptools` and
`VENV_DIR/bin/pip3 install tensorflow==2.1.0 tensorflow-addons==0.8.1
tensorflow-hub==0.7.0 tensorflow-probability==0.9.0 gym==0.15.4`.

#### Problems With Executing tensorboard

If `tensorboard` cannot be found, make sure the directory with pip installed
packages is in your PATH (that directory is either in your virtual environment
if you use a virtual environment, or it should be `~/.local/bin` on Linux
and `%UserProfile%\AppData\Roaming\Python\Python3[5-7]` and
`%UserProfile%\AppData\Roaming\Python\Python3[5-7]\Scripts` on Windows).

### Teamwork

Solving assignments in teams of size 2 or 3 is encouraged, but everyone has to
participate (it is forbidden not to work on an assignment and then submit
a solution created by other team members). All members of the team
**must** submit in ReCodEx **individually**, but can have exactly the same
sources/models/results. **Each such solution must explicitly list all
members of the team to allow plagiarism detection using
[this template](https://github.com/ufal/npfl114/tree/master/labs/team_description.py).**
