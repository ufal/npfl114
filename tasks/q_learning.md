Solve the [MountainCar-v0 environment](https://gym.openai.com/envs/MountainCar-v0)
environment from the [OpenAI Gym](https://gym.openai.com/) using the Monte Carlo
reinforcement learning algorithm. Note that this task does not require
TensorFlow.

Use the supplied [mountain_car_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/mountain_car_evaluator.py)
module to interact with the discretized environment. The environment
methods and properties are described in the `monte_carlo` assignment.
Your goal is to reach an average reward of -140 during 100 evaluation episodes.

You can start with the [q_learning.py](https://github.com/ufal/npfl114/tree/master/labs/11/q_learning.py)
template, which parses several useful parameters, creates the environment
and illustrates the overall usage.

During evaluation in ReCodEx, three different random seeds will be employed, and
you will get a point for each setting where you reach the required reward.
The time limit for each test is 5 minutes.
