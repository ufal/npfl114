The tasks are evaluated automatically using the [ReCodEx Code
Examiner](https://recodex.mff.cuni.cz/). The evaluation is
performed using Python 3.6, TensorFlow 2.0.0a0, NumPy 1.16.1
and OpenAI Gym 0.9.5.

You can install all required packages either to user packages using
`pip3 install --user tensorflow==2.0.0a0 gym==0.9.5`,
or create a virtual environment using `python3 -m venv VENV_DIR`
and then installing the packages inside it by running
`VENV_DIR/bin/pip3 install tensorflow==2.0.0a0 gym==0.9.5`.
If you have a GPU, you can install GPU-enabled TensorFlow by using
`tensorflow-gpu` instead of `tensorflow`.

### Teamwork

Working in teams of size 2 (or at most 3) is encouraged. All members of the team
must submit in ReCodEx individually, but can have exactly the same
sources/models/results. **However, each such solution must explicitly list all
members of the team to allow plagiarism detection using
[this template](https://github.com/ufal/npfl114/tree/master/labs/team_description.py).**
