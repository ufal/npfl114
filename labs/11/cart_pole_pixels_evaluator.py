#!/usr/bin/env python3
import gym
import gym.envs.classic_control
import numpy as np

class CartPolePixels(gym.envs.classic_control.CartPoleEnv):
    def __init__(self):
        super().__init__()

        self._images = 3
        self._image = np.zeros([80, 80, self._images], dtype=np.float32)
        self._viewer = None

        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(80, 80, self._images))

    def reset(self):
        observation = super().reset()
        for step in range(self._images - 1):
            self._draw(observation)
        return self._draw(observation)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self._draw(observation), reward, done, info

    def render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return

        if self._viewer is None:
            from gym.envs.classic_control import rendering
            self._viewer = rendering.SimpleImageViewer()

        self._viewer.imshow((self._image.repeat(8, axis=0).repeat(8, axis=1)*255).astype(np.uint8))

    def _draw(self, observation):
        for i in range(self._images - 1):
            self._image[:, :, i] = self._image[:, :, i + 1]
        cart = 40 + observation[0] / 3 * 40
        pole_x = int(40 + (observation[0] + np.sin(observation[2]) * 4.2) / 3 * 40)
        pole_y = int(70 - np.cos(observation[2]) * 5.2 / 3 * 40)
        self._image[:, :, self._images - 1] = 0
        self._fill_polygon([(70, cart-10), (80, cart-10), (80, cart+10), (70, cart+10)], self._image[:, :, self._images - 1], 0.5)
        self._fill_polygon([(pole_y, pole_x-2), (70, cart-2), (70, cart+2), (pole_y, pole_x+2)], self._image[:, :, self._images - 1], 1)
        return np.copy(self._image)

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

###############################
# Evaluator for NPFL1** class #
###############################

gym.envs.register(
    id="CartPolePixels-v0",
    entry_point=CartPolePixels,
    max_episode_steps=500,
    reward_threshold=475,
)

import gym_evaluator
def environment(seed=None):
    return gym_evaluator.GymEnvironment("CartPolePixels-v0", seed=seed)

# Allow running the environment and controlling it with arrows
if __name__=="__main__":
    import pyglet
    import sys
    import time

    from pyglet.window import key
    action, restart = 0, False
    def key_press(k, mod):
        global action, restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  action = 0
        if k==key.RIGHT: action = 1
    env = CartPolePixels()
    env.render()
    env._viewer.window.on_key_press = key_press
    while True:
        env.reset()
        steps, restart = 0, False
        while True:
            s, _, done, _ = env.step(action)
            steps += 1
            env.render()
            time.sleep(0.1)
            if done or restart: break
        print("Episode ended after {} steps".format(steps), file=sys.stderr)
    env.close()
