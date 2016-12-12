#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

class EnvironmentDiscrete:
    def __init__(self, env_name):
        self._env = gym.make(env_name)
        self._env_name = env_name
        if type(self._env.action_space) != gym.spaces.Discrete:
            raise ValueError("Only environments with discrete action spaces are supported!")

        if isinstance(self._env.observation_space, gym.spaces.Discrete):
            self._is_discrete = True
        elif env_name.startswith("CartPole"):
            self._is_discrete = False
            self._bins = 6
            self._separators = [
                np.linspace(-2.4, 2.4, num=self._bins + 1)[1:-1], # cart position
                np.linspace(-3, 3, num=self._bins + 1)[1:-1],     # pole angle
                np.linspace(-0.5, 0.5, num=self._bins + 1)[1:-1], # cart velocity
                np.linspace(-2, 2, num=self._bins + 1)[1:-1],     # pole angle velocity
            ]
        elif env_name.startswith("MountainCar"):
            self._is_discrete = False
            self._bins = 12
            self._separators = [
                np.linspace(-1.2, 0.6, num=self._bins + 1)[1:-1],  # car position
                np.linspace(-0.07, 0.07, num=self._bins + 1)[1:-1],# car velocity
            ]
        else:
            raise ValueError("Environment {} is not descrete and has no discretionazitaion".format(env_name))

    def _discretize(self, observation):
        if not self._is_discrete:
            buckets = np.array(observation, dtype=np.int)
            for i in range(len(observation)):
                buckets[i] = np.digitize(observation[i], self._separators[i])
            observation = np.polyval(buckets, self._bins)

        return observation

    @property
    def states(self):
        if self._is_discrete:
            return self._env.observation_space.n
        else:
            return self._bins ** len(self._separators)

    @property
    def actions(self):
        return self._env.action_space.n

    def reset(self):
        return self._discretize(self._env.reset())

    def step(self,action):
        observation, reward, done, info = self._env.step(action)

        return self._discretize(observation), reward, done, info

    def render(self):
        self._env.render()
