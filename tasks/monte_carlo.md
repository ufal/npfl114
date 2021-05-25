### Assignment: monte_carlo
#### Date: Deadline: Jun 30, 23:59
#### Points: 2 points

Solve the discretized [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment from the [OpenAI Gym](https://gym.openai.com/) using the Monte Carlo
reinforcement learning algorithm. The `gym` environments have the followng
methods and properties:
- `observation_space`: the description of environment observations
- `action_space`: the description of environment actions
- `reset() → new_state`: starts a new episode
- `step(action) → new_state, reward, done, info`: perform the chosen action
  in the environment, returning the new state, obtained reward, a boolean
  flag indicating an end of episode, and additional environment-specific
  information
- `render()`: render current environment state

We additionaly extend the `gym` environment by:
- `episode`: number of the current episode (zero-based)
- `reset(start_evaluation=False) → new_state`: if `start_evaluation` is `True`,
   an evaluation is started

Once you finish training (which you indicate by passing `start_evaluate=True`
to `reset`), your goal is to reach an average return of 475 during 100
evaluation episodes. Note that the environment prints your 100-episode
average return each 10 episodes even during training.

You can start with the [monte_carlo.py](https://github.com/ufal/npfl114/tree/master/labs/13/monte_carlo.py)
template, which parses several useful parameters, creates the environment
and illustrates the overall usage.

During evaluation in ReCodEx, three different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
