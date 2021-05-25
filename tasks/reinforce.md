### Assignment: reinforce
#### Date: Deadline: Jun 30, 23:59
#### Points: 2 points

Solve the continuous [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment from the [OpenAI Gym](https://gym.openai.com/) using the REINFORCE
algorithm. The continuous environment is very similar to the discrete one, except
that the states are vectors of real-valued observations with shape
`env.observation_space.shape`.

Your goal is to reach an average return of 475 during 100 evaluation episodes.
Start with the [reinforce.py](https://github.com/ufal/npfl114/tree/master/labs/13/reinforce.py)
template.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
