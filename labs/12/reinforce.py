#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_evaluator

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

            # Compute the action logits

            # TODO: Add a fully connected layer processing self.states, with args.hidden_layer neurons
            # and some non-linear activatin.

            # TODO: Compute `logits` using another dense layer with
            # `num_actions` outputs (utilizing no activation function).

            # TODO: Compute the `self.probabilities` from the `logits`.

            # Training

            # TODO: Compute `loss`, as a softmax cross entropy loss of self.actions and `logits`.
            # Because this is a REINFORCE algorithm, it is crucial to weight the loss of batch
            # elements using `self.returns` -- this can be accomplished using the `weights` parameter.

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.probabilities, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    evaluating = False
    while True:
        # TODO: Decide if evaluation should start (one possibility is to train for args.episodes,
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

                # TODO: Compute action distribution using `network.predict`

                # TODO: Set `action` randomly according to the generated distribution
                # (you can use np.random.choice or any other method).
                action = ...

                next_state, reward, done, _ = env.step(action)

                # TODO: Accumulate states, actions and rewards.

                state = next_state

            # TODO: Compute returns from rewards (by summing them up and
            # applying discount by `args.gamma`).

            # TODO: Extend the batch_{states,actions,returns} using the episodic
            # {states,actions,returns}.

        # TODO: Perform network training using batch_{states,actions,returns}.
