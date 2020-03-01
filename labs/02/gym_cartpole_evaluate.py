#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import gym
import numpy as np
import tensorflow as tf

def evaluate_model(model, seed=42, episodes=100, render=False, report_per_episode=False):
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """

    # Create the environment
    env = gym.make("CartPole-v1")
    env.seed(seed)

    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation = env.reset()
        score = 0
        done = False

        while not done:
            if render:
                env.render()

            prediction = model.predict_on_batch(observation[np.newaxis, ...])[0].numpy()
            if len(prediction) == 1:
                action = prediction[0] >= 0.5
            elif len(prediction) == 2:
                action = np.argmax(prediction)
            else:
                raise ValueError("Unknown model output shape, only 1 or 2 outputs are supported")

            observation, reward, done, info = env.step(action)
            score += reward

        total_score += score
        if report_per_episode:
            print("The episode {} finished with score {}.".format(episode + 1, score))
    return total_score / episodes


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="gym_cartpole_model.h5", nargs="?", type=str, help="Name of model.")
    parser.add_argument("--episodes", default=100, type=int, help="Number of episodes.")
    parser.add_argument("--render", default=False, action="store_true", help="Render the environment.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load the model
    model = tf.keras.models.load_model(args.model, compile=False)

    total_score = evaluate_model(model, args.seed, args.episodes, args.render, report_per_episode=True)
    print("The average reward per episode was {:.2f}.".format(total_score))
