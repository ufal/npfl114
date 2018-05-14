Solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment from the [OpenAI Gym](https://gym.openai.com/) using the Monte Carlo
reinforcement learning algorithm. Note that this task does not require
TensorFlow.

Use the supplied [cart_pole_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/cart_pole_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/gym_evaluator.py)
to interact with the discretized environment. The environment has the
following methods and properties:
- `states`: number of states of the environment
- `actions`: number of actions of the environment
- `reset(start_evaluate=False) → new_state`: starts a new episode
- `step(action) → new_state, reward, done, info`: perform the chosen action
  in the environment, returning the new state, obtained reward, a boolean
  flag indicating an end of episode, and additional environment-specific
  information
- `render()`: render current environment state

Once you finish training (which you indicate by passing `start_evaluate=True`
to `reset`), your goal is to reach an average reward of 475 during 100
evaluation episodes. Note that the environment prints your 100-episode
average reward each 10 episodes even during training.

You can start with the [monte_carlo.py](https://github.com/ufal/npfl114/tree/master/labs/11/monte_carlo.py)
template, which parses several useful parameters, creates the environment
and illustrates the overall usage.

During evaluation in ReCodEx, three different random seeds will be employed, and
you will get a point for each setting where you reach the required reward.
The time limit for each test is 5 minutes.
