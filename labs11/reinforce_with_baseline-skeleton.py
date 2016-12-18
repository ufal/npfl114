#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import environment_continuous
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

class PolicyGradientWithBaseline:
    def __init__(self, observations, policy_and_value_network, learning_rate, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        # Construct the graph
        with self.session.graph.as_default():
            # TODO: define the following, using policy_and_value_network
            # logits = ...
            # self.value = ...
            # self.probabilities = ... [probabilities of all actions]

            self.chosen_actions = tf.placeholder(tf.int32, [None])
            self.returns = tf.placeholder(tf.float32, [None])

            # TODO: compute loss of both the policy and value
            # loss_policy = ... [cross_entropy between logits and chosen_actions, multiplying it by (self.returns - self.value)]
            # loss_value = ... [MSE of self.return and self.value]
            # self.training = ... [use loss_policy + loss_value, and use learning_rate]

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    def predict(self, observations):
        return self.session.run(self.probabilities,
                                {self.observations: observations})

    def train(self, observations, chosen_actions, returns):
        self.session.run(self.training,
                         {self.observations: observations,
                          self.chosen_actions: chosen_actions,
                          self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Name of the environment.")
    parser.add_argument("--episodes", default=1000, type=int, help="Episodes in a batch.")
    parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of steps.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--alpha", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of episodes to train on.")
    parser.add_argument("--hidden_layer", default=20, type=int, help="Size of hidden layer.")
    args = parser.parse_args()

    # Create the environment
    env = environment_continuous.EnvironmentContinuous(args.env)
    if args.render_each:
        # Because of low TLS limit, load OpenGL before TensorFlow
        env.reset()
        env.render()

    # Create policy and value network
    def policy_and_value_network(observations):
        # TODO: return logits, value
        # logits are computed from observations using a NN with a single hidden layer of size args.hidden_layer and output layer of size env.actions
        # value is computed from observations using a NN with another single hidden layer of size args.hidden_layer and one output
        # logits = ...
        # value = ...
        return logits, value

    pg = PolicyGradientWithBaseline(observations=env.observations, policy_and_value_network=policy_and_value_network,
                                    learning_rate=args.alpha, threads=args.threads)

    episode_returns, episode_lengths = [], []
    for batch_start in range(0, args.episodes, args.batch_size):
        # Collect data for training
        observations, actions, rewards = [], [], []
        for episode in range(batch_start, batch_start + args.batch_size):
            # Perform episode
            observation = env.reset()
            total_reward = 0
            for t in range(args.max_steps):
                if args.render_each and episode > 0 and episode % args.render_each == 0:
                    env.render()

                # TODO: compute probabilities and value function using pg.predict, and choose action according to the probabilities
                # probabilities = ...
                # value = ...
                # action = ...

                observations.append(observation)
                actions.append(action)

                # perform step in the environment
                observation, reward, done, _ = env.step(action)

                total_reward += reward
                rewards.append(reward)

                if done:
                    break

            # TODO: sum and discount rewards, only the last t of them

            episode_returns.append(total_reward)
            episode_lengths.append(t)
            if len(episode_returns) % 10 == 0:
                # Evaluate
                observation, total_reward = env.reset(), 0
                for i in range(args.max_steps):
                    [probabilities], _ = pg.predict([observation])
                    action = np.random.choice(np.arange(len(probabilities)), p=probabilities)
                    observation, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break

                print("Episode {}, current evaluation reward {}, mean 100-episode reward {}, mean 100-episode length {}.".format(
                    episode + 1, total_reward, np.mean(episode_returns[-100:]), np.mean(episode_lengths[-100:])))

        pg.train(observations, actions, rewards)
