#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import mountain_car_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_states, num_actions):
        with self.session.graph.as_default():
            # Input states
            self.states = tf.placeholder(tf.int32, [None])
            # Input q_values (uses as targets for training)
            self.q_values = tf.placeholder(tf.float32, [None, num_actions])

            # TODO: Compute one-hot representation of self.states.

            # TODO: Compute the q_values as a single fully connected layer without activation,
            # with `num_actions` outputs, using the one-hot encoded states. It is important
            # to use such trivial architecture for the network to train at all.

            # Training
            # TODO: Perform the training, using mean squared error of the given
            # `q_values` and the predicted ones.

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        # TODO: Predict q_values for given states

    def train(self, states, q_values):
        # TODO: Given states and target Q-values, perform the training

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(discrete=True)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.states, env.actions)

    evaluating = False
    epsilon = args.epsilon
    while True:
        # TODO: decide if we want to start evaluating -- maybe after already processing
        # args.episodes (i.e., env.episode >= args.episodes), but you can use other logis.

        # Perform episode
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: compute q_values using the network and action using epsilon-greedy policy.
            action = ...

            next_state, reward, done, _ = env.step(action)

            # Perform the network update

            # TODO: Compute the q_values of the next_state

            # TODO: Update the goal q_values for the state `state`, using the TD update
            # for action `action` (leaving the q_values for different actions unchanged).

            # TODO: Train the network using the computed goal q_values for state `state`.

            state = next_state

        # Epsilon interpolation
        if args.epsilon_final:
            epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
