### Assignment: reinforce
#### Date: Deadline: Jun 30, 23:59
#### Points: 2 points

Solve the continuous [CartPole-v1 environment](https://www.gymlibrary.ml/environments/classic_control/cart_pole/)
environment from the [Gym library](https://www.gymlibrary.ml/) using the REINFORCE
algorithm. The continuous environment is very similar to the discrete one, except
that the states are vectors of real-valued observations with shape
`env.observation_space.shape`.

Your goal is to reach an average return of 475 during 100 evaluation episodes.

Start with the [reinforce.py](https://github.com/ufal/npfl114/tree/past-2122/labs/13/reinforce.py)
template, which provides a simple network implementation in TensorFlow. However,
feel free to use PyTorch instead, if you like.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
