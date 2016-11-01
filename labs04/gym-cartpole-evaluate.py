from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf

class Network:
    def __init__(self, path, threads):
        # Create the session
        self.session = tf.Session(graph = tf.Graph(), config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                            intra_op_parallelism_threads=threads))

        # Load the metagraph
        with self.session.graph.as_default():
            self.saver = tf.train.import_meta_graph(path + ".meta")

            # Attach the end points
            self.observations = tf.get_collection("end_points/observations")[0]
            self.action = tf.get_collection("end_points/action")[0]

        # Load the graph weights
        self.saver.restore(self.session, path)

    def predict(self, observations):
        return self.session.run(self.action, {self.observations: [observations]})[0]


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tensorflow_model", type=str, help="Name of tensorflow model.")
    parser.add_argument("--episodes", default=100, type=int, help="Number of episodes.")
    parser.add_argument("--render", dest="render", action="store_true", help="Render the environment.")
    parser.set_defaults(render=False)
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = gym.make('CartPole-v1')

    # Load the network
    network = Network(args.tensorflow_model, args.threads)

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
