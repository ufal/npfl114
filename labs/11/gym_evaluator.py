#!/usr/bin/env python3
import sys

import gym
import numpy as np

class GymEnvironment:
    def __init__(self, env, bins=None, separators=None):
        self._env = gym.make(env)
        self._env.seed(42)

        self._bins = bins
        self._separators = separators

        self._evaluating_from = None
        self._episode_reward = 0
        self._episode_rewards = []

    def _maybe_discretize(self, observation):
        if self._bins is not None:
            buckets = np.array(observation, dtype=np.int)
            for i in range(len(observation)):
                buckets[i] = np.digitize(observation[i], self._separators[i])
            observation = np.polyval(buckets, self._bins)

        return observation

    @property
    def states(self):
        if self._bins is not None:
            return self._bins ** len(self._separators)
        raise RuntimeError("Continuous environments have infinitely many states")

    @property
    def state_shape(self):
        if self._bins is not None:
            return []
        else:
            return list(self._env.observation_space.shape)

    @property
    def actions(self):
        return self._env.action_space.n

    @property
    def episode(self):
        return len(self._episode_rewards)

    def reset(self, start_evaluate=False):
        if start_evaluate and self._evaluating_from is None:
            self._evaluating_from = self.episode

        return self._maybe_discretize(self._env.reset())

    def step(self, action):
        observation, reward, done, info = self._env.step(action)

        self._episode_reward += reward
        if done:
            self._episode_rewards.append(self._episode_reward)

            if self.episode % 10 == 0:
                print("Episode {}, mean 100-episode reward {}".format(
                    self.episode, np.mean(self._episode_rewards[-100:])), file=sys.stderr)
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + 100:
                print("The mean 100-episode reward after evaluation {}".format(np.mean(self._episode_rewards[-100:])))
                sys.exit(0)

            self._episode_reward = 0

        return self._maybe_discretize(observation), reward, done, info

    def render(self):
        self._env.render()
