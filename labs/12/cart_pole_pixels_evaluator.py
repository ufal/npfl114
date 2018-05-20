#!/usr/bin/env python3
import sys

import gym
import numpy as np

class CartPolePixelsEvaluator:
    def __init__(self):
        self._env = gym.make("CartPole-v1")
        self._env.seed(42)
        self._iw = None

        self._images = 3
        self._image = np.zeros([80, 80, self._images], dtype=np.float32)

        self._evaluating_from = None
        self._episode_reward = 0
        self._episode_rewards = []

    @property
    def states(self):
        raise RuntimeError("Continuous environments have infinitely many states")

    @property
    def state_shape(self):
        return list(self._image.shape)

    @property
    def actions(self):
        return self._env.action_space.n

    @property
    def episode(self):
        return len(self._episode_rewards)

    def reset(self):
        observation = self._env.reset()
        for step in range(self._images - 1):
            self._draw(observation)
        return self._draw(observation)

    def reset(self, start_evaluate=False):
        if start_evaluate and self._evaluating_from is None:
            self._evaluating_from = self.episode

        observation = self._env.reset()
        for step in range(self._images - 1):
            self._draw(observation)
        return self._draw(observation)

    def step(self,action):
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

        return self._draw(observation), reward, done, info

    def render(self):
        if self._iw is None:
            import gym.envs.classic_control.rendering as gym_rendering
            self._iw = gym_rendering.SimpleImageViewer()

        self._iw.imshow((self._image[:, :, :3]*255).astype(np.uint8))

    def _draw(self, observation):
        for i in range(self._images - 1):
            self._image[:, :, i] = self._image[:, :, i + 1]
        cart = 40 + observation[0] / 3 * 40
        pole_x = int(40 + (observation[0] + np.sin(observation[2]) * 4.2) / 3 * 40)
        pole_y = int(70 - np.cos(observation[2]) * 5.2 / 3 * 40)
        self._image[:, :, self._images - 1] = 0
        self._fill_polygon([(70, cart-10), (80, cart-10), (80, cart+10), (70, cart+10)], self._image[:, :, self._images - 1], 0.5)
        self._fill_polygon([(pole_y, pole_x-2), (70, cart-2), (70, cart+2), (pole_y, pole_x+2)], self._image[:, :, self._images - 1], 1)
        return self._image

    # Taken from https://github.com/luispedro/mahotas/blob/master/mahotas/polygon.py
    def _fill_polygon(self, polygon, canvas, color=1):
        '''
        fill_polygon([(y0,x0), (y1,x1),...], canvas, color=1)
        Draw a filled polygon in canvas
        Parameters
        ----------
        polygon : list of pairs
            a list of (y,x) points
        canvas : ndarray
            where to draw, will be modified in place
        color : integer, optional
            which colour to use (default: 1)
        '''
        # algorithm adapted from: http://www.alienryderflex.com/polygon_fill/
        if not len(polygon):
            return
        min_y = min(y for y,x in polygon)
        max_y = max(y for y,x in polygon)
        polygon = [(float(y),float(x)) for y,x in polygon]
        if max_y < canvas.shape[0]:
            max_y += 1
        for y in range(min_y, max_y):
            nodes = []
            j = -1
            for i,p in enumerate(polygon):
                pj = polygon[j]
                if p[0] < y and pj[0] >= y or pj[0] < y and p[0] >= y:
                    dy = pj[0] - p[0]
                    if dy:
                        nodes.append( (p[1] + (y-p[0])/(pj[0]-p[0])*(pj[1]-p[1])) )
                    elif p[0] == y:
                        nodes.append(p[1])
                j = i
            nodes.sort()
            for n,nn in zip(nodes[::2],nodes[1::2]):
                nn += 1
                canvas[y, int(n):int(nn)] = color

def environment():
    return CartPolePixelsEvaluator()
