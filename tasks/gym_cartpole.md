Solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
from the [OpenAI Gym](https://gym.openai.com/), utilizing only provided supervised
training data. The data is available in
[gym_cartpole-data.txt](https://github.com/ufal/npfl114/tree/master/labs/02/gym_cartpole-data.txt)
file, each line containing one observation (four space separated floats) and
a corresponding action (the last space separated integer). Start with the
[gym_cartpole.py](https://github.com/ufal/npfl114/tree/master/labs/02/gym_cartpole.py).

The solution to this task should be a _model_ which passes evaluation on random
inputs. This evaluation is performed by running the
[gym_cartpole_evaluate.py](https://github.com/ufal/npfl114/tree/master/labs/02/gym_cartpole_evaluate.py),
which loads a model and then evaluates it on 100 random episodes (optionally
rendering if `--render` option is provided). In order to pass, you must achieve
an average reward of at least 475 on 100 episodes.

_The size of the training data is very small and you should consider
it when designing the model._

To submit your model in ReCodEx, use the supplied
[gym_cartpole_recodex.py](https://github.com/ufal/npfl114/tree/master/labs/02/gym_cartpole_recodex.py)
script. When executed, the script embeds the saved model in current
directory into a script `gym_cartpole_recodex_submission.py`, which can
be submitted in ReCodEx. Note that by default there are at most
**five** submission attempts, write me if you need more.
