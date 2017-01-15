#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import gym
import gym.envs.classic_control.rendering as gym_rendering
import numpy as np

class EnvironmentPixels:
    def __init__(self, env_name):
        self._env_name = env_name
        self._env = gym.make(env_name)
        self._iw = gym_rendering.SimpleImageViewer()
        if self._env_name.startswith("CartPole"):
            self._images = 3
            self._image = np.zeros([80, 80, self._images], dtype=np.float32)
        else:
            raise ValueError("Environment {} not supported in EnvironmentPixels".format(env_name))

    def _draw(self, observation):
        if self._env_name.startswith("CartPole"):
            for i in range(self._images - 1):
                self._image[:, :, i] = self._image[:, :, i + 1]
            cart = 40 + observation[0] / 3 * 40
            pole_x = int(40 + (observation[0] + np.sin(observation[2]) * 4.2) / 3 * 40)
            pole_y = int(70 - np.cos(observation[2]) * 5.2 / 3 * 40)
            self._image[:, :, self._images - 1] = 0
            self._fill_polygon([(70, cart-10), (80, cart-10), (80, cart+10), (70, cart+10)], self._image[:, :, self._images - 1], 0.5)
            self._fill_polygon([(pole_y, pole_x-2), (70, cart-2), (70, cart+2), (pole_y, pole_x+2)], self._image[:, :, self._images - 1], 1)
            return np.copy(self._image)

    @property
    def observations(self):
        return list(self._image.shape)

    @property
    def actions(self):
        return self._env.action_space.n

    def reset(self):
        observation = self._env.reset()
        for step in range(self._images - 1):
            self._draw(observation)
        return self._draw(observation)

    def step(self,action):
        observation, reward, done, info = self._env.step(action)

        return self._draw(observation), reward, done, info

    def render(self):
        if self._env_name.startswith("CartPole"):
            self._iw.imshow((self._image[:, :, :3]*255).astype(np.uint8))

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

