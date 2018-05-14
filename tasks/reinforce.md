Solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment from the [OpenAI Gym](https://gym.openai.com/) using the REINFORCE
algorithm.

Use the supplied [cart_pole_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/cart_pole_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/gym_evaluator.py)
to interact with the _continuous_ environment. The environment
has the same properties and methods as the discrete environment described
in `monte_carlo` task, with the following additions:
- the continuous environment has to be created with `discrete=False` option
- `state_shape`: the shape describing the floating point state tensor
- `states`: as the number of states is infinite, raises an exception

Once you finish training (which you indicate by passing `start_evaluate=True`
to `reset`), your goal is to reach an average reward of 475 during 100
evaluation episodes. Note that the environment prints your 100-episode
average reward each 10 episodes even during training. You should start with the
[reinforce.py](https://github.com/ufal/npfl114/tree/master/labs/12/reinforce.py)
template.

During evaluation in ReCodEx, two different random seeds will be employed, and
you will get a point for each setting where you reach the required reward.
The time limit for each test is 5 minutes.

After solving this task, you should continue with `reinforce_with_baseline`.
