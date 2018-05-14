Solve the [MountainCar-v0 environment](https://gym.openai.com/envs/MountainCar-v0)
environment from the [OpenAI Gym](https://gym.openai.com/) using a Q-network
(neural network variant of Q-learning algorithm).

Note that training DQN (Deep Q-Networks) is inherently tricky and unstable.
Therefore, we will implement a direct analogue of tabular Q-learning, allowing
the network to employ independent weights for every discretized environment state.

Use the supplied [mountain_car_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/mountain_car_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/gym_evaluator.py)
to interact with the discretized environment. The environment
methods and properties are described in the `monte_carlo` assignment.
Your goal is to reach an average reward of -200 during 100 evaluation episodes.

You can start with the [q_network.py](https://github.com/ufal/npfl114/tree/master/labs/12/q_network.py)
template. Note that setting hyperparameters of
Q-network is even more tricky than for Q-learning – if you try to vary the
architecture, it might not learn at all.

During evaluation in ReCodEx, two different random seeds will be employed, and
you will get a point for each setting where you reach the required reward.
The time limit for each test is 10 minutes.
