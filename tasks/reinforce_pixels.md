### Assignment: reinforce_pixels
#### Date: Deadline: Jun 30, 23:59
#### Points: 2 points

This is a continuation of the `reinforce` or `reinforce_baseline` assignments.

The supplied [cart_pole_pixels_environment.py](https://github.com/ufal/npfl114/tree/master/labs/13/cart_pole_pixels_environment.py)
generates a pixel representation of the `CartPole` environment
as an $80×80$ image with three channels, with each channel representing one time step
(i.e., the current observation and the two previous ones).

To pass the assignment, you need to reach an average return of 400 in 100
evaluation episodes. During evaluation in ReCodEx, two different random seeds
will be employed, and you need to reach the required return on all of them. Time
limit for each test is 10 minutes.

You should probably train the model locally and submit the already pretrained
model to ReCodEx.

You can start with the
[reinforce_pixels.py](https://github.com/ufal/npfl114/tree/master/labs/13/reinforce_pixels.py)
template using the correct environment.
