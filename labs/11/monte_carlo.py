#!/usr/bin/env python3
import numpy as np

import cart_pole_evaluator

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(42)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=True)

    # Create Q, C and other variables
    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [env.states, env.actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with shape [env.states, env.actions],
    #   representing number of observed returns of a given (state, action) pair.

    for _ in range(args.episodes):
        # Perform episode
        state = env.reset()
        states, actions, rewards = [], [], []
        while True:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of args.epsilon, use a random actions (there are env.actions of them),
            # otherwise, choose and action with maximum Q[state, action].

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # TODO: Compute returns from the observed rewards.

        # TODO: Update Q and C

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
