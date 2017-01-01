#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import environment_continuous
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import threading

class AsynchronousActorCritic:
    def __init__(self, observations, policy_and_value_network, learning_rate, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        # Construct the graph
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, observations])
            # TODO: define the following, using policy_and_value_network
            # logits = ...
            # self.value = ...
            # self.probabilities = ... [probabilities of all actions]

            self.chosen_actions = tf.placeholder(tf.int32, [None])
            self.returns = tf.placeholder(tf.float32, [None])

            # TODO: compute loss of both the policy and value
            # loss_policy = ... [cross_entropy between logits and chosen_actions,
            #                    multiplied by (self.returns - tf.stop_gradient(self.value))]
            # loss_value = ... [MSE of self.return and self.value]

            # Training with gradient clipping to 10
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grad_and_vars = optimizer.compute_gradients(loss_policy + loss_value)
            vars = [var for grad, var in grad_and_vars]
            grads = [grad for grad, var in grad_and_vars]
            grads, _ = tf.clip_by_global_norm(grads, 10)
            self.training = optimizer.apply_gradients(zip(grads, vars))

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    def predict(self, observations):
        return self.session.run(self.probabilities,
                                {self.observations: observations})

    def predict_value(self, observations):
        return self.session.run(self.value,
                                {self.observations: observations})

    def train(self, observations, chosen_actions, returns):
        self.session.run(self.training,
                         {self.observations: observations,
                          self.chosen_actions: chosen_actions,
                          self.returns: returns})

class Agent:
    def __init__(self, a3c, env_name, max_steps, n_steps, gamma):
        self.a3c = a3c
        self.env = environment_continuous.EnvironmentContinuous(env_name)
        self.max_steps = max_steps
        self.n_steps = n_steps
        self.gamma = gamma

        self.start()

    def start(self):
        self.steps, self.episode_return = 0, 0
        self.observations, self.actions, self.rewards = [self.env.reset()], [], []

    def step(self):
        # TODO: Perform one step, save action, observation and reward, update steps and episode_return

        # TODO: Finish if self.steps >= self.max_steps

        # TODO: If finished or after n_steps, perform training.
        # Compute predicted value in last state using a3c.predict_value, compute returns,
        # perform training

        return done, self.episode_return, self.steps

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Name of the environment.")
    parser.add_argument("--episodes", default=10000, type=int, help="Episodes in a batch.")
    parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of steps.")
    parser.add_argument("--threads", default=1, type=int, help="Number of threads.")
    parser.add_argument("--agents", default=1, type=int, help="Number of agents per thread.")

    parser.add_argument("--alpha", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--n_steps", default=20, type=int, help="Number of actions to train on.")
    args = parser.parse_args()

    # Create the environment
    env = environment_continuous.EnvironmentContinuous(args.env)

    # Create policy and value network
    def policy_and_value_network(observations):
        # TODO: Example network, you may choose another
        hidden_layer = tf_layers.fully_connected(observations, 200, activation_fn=tf.nn.relu)
        hidden_layer = tf_layers.fully_connected(hidden_layer, 100, activation_fn=tf.nn.relu)
        logits = tf_layers.linear(hidden_layer, env.actions)
        value = tf_layers.linear(hidden_layer, 1)
        return logits, value

    a3c = AsynchronousActorCritic(observations=env.observations, policy_and_value_network=policy_and_value_network,
                                  learning_rate=args.alpha, threads=2)

    episodes, episodes_return, episodes_length = 0, [], []
    episodes_lock = threading.Lock()
    def worker_thread():
        global episodes
        agents = []
        for i in range(args.agents):
            agents.append(Agent(a3c, args.env, args.max_steps, args.n_steps, args.gamma))

        while agents:
            for i in reversed(range(len(agents))):
                done, episode_return, episode_length = agents[i].step()

                if done:
                    with episodes_lock:
                        episodes_return.append(episode_return)
                        episodes_length.append(episode_length)
                        episodes += 1
                        episode = episodes

                    if episode >= args.episodes:
                        del agents[i]
                        continue

                    if episode % 10 == 0:
                        print("Episode {}, mean 100-episode reward {}, mean 100-episode length {}.".format(
                            episode, np.mean(episodes_return[-100:]), np.mean(episodes_length[-100:])))

                    agents[i].start()

    threads = []
    for i in range(args.threads):
        threads.append(threading.Thread(target=worker_thread))
    for thread in threads:
        thread.daemon = True
        thread.start()
    for thread in threads:
        while thread.is_alive():
            thread.join(1)

    print("All finished, mean 100-episode reward {}, mean 100-episode length {}.".format(
        np.mean(episodes_return[-100:]), np.mean(episodes_length[-100:])))
