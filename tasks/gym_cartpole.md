Solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
from the [OpenAI Gym](https://gym.openai.com/), utilizing only provided supervised
training data. The data is available in
[gym_cartpole-data.txt](https://github.com/ufal/npfl114/tree/master/labs/02/gym_cartpole-data.txt)
file, each line containing one observation (four space separated floats) and
a corresponding action (the last space separated integer). Start with the
[gym_cartpole.py](https://github.com/ufal/npfl114/tree/master/labs/02/gym_cartpole.py).

The solution to this task should be a _model_ which passes evaluation on random
inputs. This evaluation is performed by running the
[gym_cartpole-evaluate.py](https://github.com/ufal/npfl114/tree/master/labs/02/gym_cartpole-evaluate.py),
which loads a model and then evaluates it on 100 random episodes (optionally
rendering if `--render` option is provided). In order to pass, you must achieve
an average reward of at least 475 on 100 episodes.

_The size of the training data is very small and you should consider
it when designing the model._

**Note that this task cannot be submitted in ReCodEx yet, it will be added in
a week. Until that time, you can perform local evaluation by yourselves.**
