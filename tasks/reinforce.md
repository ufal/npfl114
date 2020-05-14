### Assignment: reinforce
#### Date: Deadline: May 24, 23:59
#### Points: 2 points

Solve the continuous [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment from the [OpenAI Gym](https://gym.openai.com/) using the REINFORCE
algorithm.

The supplied [cart_pole_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/cart_pole_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/gym_evaluator.py))
can create a continuous environment using `environment(discrete=False)`.
The continuous environment is very similar to the discrete environment, except
that the states are vectors of real-valued observations with shape `environment.state_shape`.

Your goal is to reach an average return of 475 during 100 evaluation episodes.
Start with the [reinforce.py](https://github.com/ufal/npfl114/tree/master/labs/11/reinforce.py)
template.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.

**Note that `gym_evaluator.py` and `cart_pole_evaluator.py` must not be submitted to ReCodEx.**
