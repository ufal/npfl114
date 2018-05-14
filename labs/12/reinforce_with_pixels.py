#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_pixels_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            # Input states
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            # Chosen actions (used for training)
            self.actions = tf.placeholder(tf.int32, [None])
            # Observed returns (used for training)
            self.returns = tf.placeholder(tf.float32, [None])

            # TODO: Compute the `logits`, `self.probabilities` and `baseline`.

            # Training

            # TODO: As in `reinforce_with_baseline`, compute the `loss_softmax` and `loss_baseline`.

            global_step = tf.train.create_global_step()
            optimizer =  tf.train.AdamOptimizer(args.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss_softmax + loss_baseline))
            gradient_norm = tf.global_norm(gradients)
            if args.clip_gradient:
                gradients, _ = tf.clip_by_global_norm(gradients, args.clip_gradient, gradient_norm)
            self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

            # Summaries
            loss_l2 = tf.add_n([tf.nn.l2_loss(variable) for variable in tf.trainable_variables()])
            logit_entropy = tf.reduce_mean(tf.distributions.Categorical(logits=logits).entropy())

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.summaries = [
                    tf.contrib.summary.scalar("train/loss", loss_softmax + loss_baseline),
                    tf.contrib.summary.scalar("train/loss_softmax", loss_softmax),
                    tf.contrib.summary.scalar("train/loss_baseline", loss_baseline),
                    tf.contrib.summary.scalar("train/loss_l2", loss_l2),
                    tf.contrib.summary.scalar("train/return", tf.reduce_mean(self.returns)),
                    tf.contrib.summary.scalar("train/baseline", tf.reduce_mean(baseline)),
                    tf.contrib.summary.scalar("train/return-baseline", tf.reduce_mean(self.returns - baseline)),
                    tf.contrib.summary.scalar("train/gradient_norm", gradient_norm),
                    tf.contrib.summary.scalar("train/logit_entropy", logit_entropy),
                    tf.contrib.summary.histogram("train/probabilities", self.probabilities),
                ]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def predict(self, states):
        return self.session.run(self.probabilities, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes to train on.")
    parser.add_argument("--clip_gradient", default=None, type=float, help="Gradient clipping.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    evaluating = False
    while True:
        # TODO(reinforce): Decide if evaluation should start (one possibility is to train for args.episodes,
        # so env.episode >= args.episodes could be used).
        evaluation = ...

        # Train for a batch of episodes
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            state = env.reset(evaluating)
            states, actions, rewards, done = [], [], [], False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # TODO(reinforce): Compute action distribution using `network.predict`

                # TODO(reinforce): Set `action` randomly according to the generated distribution
                # (you can use np.random.choice or any other method).
                action = ...

                next_state, reward, done, _ = env.step(action)

                # TODO(reinforce): Accumulate states, actions and rewards.

                state = next_state

            # TODO(reinforce): Compute returns from rewards (by summing them up and
            # applying discount by `args.gamma`).

            # TODO(reinforce): Extend the batch_{states,actions,returns} using the episodic
            # {states,actions,returns}.

        # TODO(reinforce): Perform network training using batch_{states,actions,returns}.
