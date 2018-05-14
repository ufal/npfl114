This is a continuation of `reinforce` assignment.

Using the [reinforce_with_baseline.py](https://github.com/ufal/npfl114/tree/master/labs/12/reinforce_with_baseline.py)
template, modify the REINFORCE algorithm to use a baseline.

Using a baseline lowers the variance of the value function gradient estimator,
which allows faster training and decreases sensitivity to hyperparameter values.
To reflect this effect in ReCodEx, note that the evaluation phase will
_automatically start after 200 episodes_. Using only 200 episodes for training
in this setting is probably too little for the REINFORCE algorithm, but
suffices for the variant with a baseline.

During evaluation in ReCodEx, two different random seeds will be employed, and
you will get a point for each setting where you reach the required reward.
The time limit for each test is 5 minutes.
