#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Optional
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

def evaluate_model(
    model: tf.keras.Model, seed:int = 42, episodes:int = 100, render:bool = False, report_per_episode:bool = False
) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gym

    # Create the environment
    env = gym.make("CartPole-v1")
    env.seed(seed)

    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset(), 0, False
        while not done:
            if render:
                env.render()

            prediction = model(observation[np.newaxis, ...])[0].numpy()
            if len(prediction) == 1:
                action = 1 if prediction[0] >= 0.5 else 0
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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate the given model")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--render", default=False, action="store_true", help="Render during evaluation")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
parser.add_argument("--model", default="gym_cartpole_model.h5", type=str, help="Output model path.")

def main(args: argparse.Namespace) -> Optional[tf.keras.Model]:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if not args.evaluate:
        # Create logdir name
        args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
        ))

        # Load the data
        data = np.loadtxt("gym_cartpole_data.txt")
        observations, labels = data[:, :-1], data[:, -1].astype(np.int32)

        # TODO: Create the model in the `model` variable. Note that
        # the model can perform any of:
        # - binary classification with 1 output and sigmoid activation;
        # - two-class classification with 2 outputs and softmax activation.
        model = ...

        # TODO: Prepare the model for training using the `model.compile` method.
        model.compile(...)

        tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        model.fit(
            observations, labels,
            batch_size=args.batch_size, epochs=args.epochs,
            callbacks=[tb_callback]
        )

        # Save the model, without the optimizer state.
        model.save(args.model, include_optimizer=False)

    else:
        # Evaluating, either manually or in ReCodEx
        model = tf.keras.models.load_model(args.model)

        if args.recodex:
            return model
        else:
            score = evaluate_model(model, seed=args.seed, render=args.render, report_per_episode=True)
            print("The average score was {}.".format(score))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
