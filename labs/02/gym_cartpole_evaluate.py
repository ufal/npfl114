#!/usr/bin/env python3
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.summary # Needed to allow importing summary operations

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))
    def load(self, path):
        # Load the metagraph
        with self.session.graph.as_default():
            self.saver = tf.train.import_meta_graph(path + ".meta")

            # Attach the end points
            self.observations = tf.get_collection("end_points/observations")[0]
            self.actions = tf.get_collection("end_points/actions")[0]

        # Load the graph weights
        self.saver.restore(self.session, path)

    def predict(self, observations):
        return self.session.run(self.actions, {self.observations: [observations]})[0]


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="gym_cartpole/model", nargs="?", type=str, help="Name of tensorflow model.")
    parser.add_argument("--episodes", default=100, type=int, help="Number of episodes.")
    parser.add_argument("--render", default=False, action="store_true", help="Render the environment.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = gym.make("CartPole-v1")

    # Construct and load the network
    network = Network(threads=args.threads)
    network.load(args.model)

    # Evaluate the episodes
    total_score = 0
    for episode in range(args.episodes):
        observation = env.reset()
        score = 0
        for i in range(env.spec.timestep_limit):
            if args.render:
                env.render()
            observation, reward, done, info = env.step(network.predict(observation))
            score += reward
            if done:
                break

        total_score += score
        print("The episode {} finished with score {}.".format(episode + 1, score))

    print("The average reward per episode was {:.2f}.".format(total_score / args.episodes))
