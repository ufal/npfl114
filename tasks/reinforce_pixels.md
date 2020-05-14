### Assignment: reinforce_pixels
#### Date: Deadline: May 24, 23:59
#### Points: 2 points

This is a continuation of the `reinforce` or `reinforce_baseline` assignments.

The supplied [cart_pole_pixels_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/cart_pole_pixels_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/gym_evaluator.py))
generates a pixel representation of the `CartPole` environment
as an $80×80$ image with three channels, with each channel representing one time step
(i.e., the current observation and the two previous ones).

To pass the assignment, you need to reach an average return of 250 during 100
evaluation episodes. During evaluation in ReCodEx, two different random seeds
will be employed, and you need to reach the required return on all of them. Time
limit for each test is 10 minutes.

You can start with the
[reinforce_pixels.py](https://github.com/ufal/npfl114/tree/master/labs/11/reinforce_pixels.py)
template using the correct environment.

**Note that `gym_evaluator.py` and `cart_pole_pixels_evaluator.py` must not be submitted to ReCodEx.**
