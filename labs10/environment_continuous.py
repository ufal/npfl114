#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

class EnvironmentContinuous:
    def __init__(self, env_name):
        self._env = gym.make(env_name)
        self._env_name = env_name
        if type(self._env.action_space) != gym.spaces.Discrete:
            raise ValueError("Only environments with discrete action spaces are supported!")

        self._is_discrete = isinstance(self._env.observation_space, gym.spaces.Discrete)

    def _continuize(self, observation):
        if self._is_discrete:
            one_hot = np.zeros([self._env.observation_space.n], dtype=np.float32)
            one_hot[observation] = 1
            observation = one_hot

        return observation

    @property
    def observations(self):
        if self._is_discrete:
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape[0]

    @property
    def actions(self):
        return self._env.action_space.n

    def reset(self):
        return self._continuize(self._env.reset())

    def step(self,action):
        observation, reward, done, info = self._env.step(action)

        return self._continuize(observation), reward, done, info

    def render(self):
        self._env.render()
