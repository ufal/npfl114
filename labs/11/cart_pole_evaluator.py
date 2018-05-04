#!/usr/bin/env python3
import numpy as np

import gym_discrete_evaluator

def environment():
    bins = 6
    separators = [
        np.linspace(-2.4, 2.4, num=bins + 1)[1:-1], # cart position
        np.linspace(-3, 3, num=bins + 1)[1:-1],     # pole angle
        np.linspace(-0.5, 0.5, num=bins + 1)[1:-1], # cart velocity
        np.linspace(-2, 2, num=bins + 1)[1:-1],     # pole angle velocity
    ]

    return gym_discrete_evaluator.GymEnvironment("CartPole-v1", bins, separators)
