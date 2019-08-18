"""
This is the main application entry point. It runs the cartpole game
via the cartpole_dqn and cartpole_env modules.
"""

import numpy as np
import sys, termios, tty, os
from cartpole_env import CartPoleEnv
from cartpole_dqn import CartpoleDQN
import matplotlib.pyplot as plt

class Cartpole():

    """
    Cartpole runs the game using the deep neural network and the OpenAI Gym
    """

    IMITATION_MODE = False

    USER_INPUT = dict()
    USER_INPUT[0] = "APPLY FORCE LEFT"
    USER_INPUT[1] = "APPLY FORCE RIGHT"
    USER_INPUT[2] = "EXIT"
    USER_INPUT[3] = "NO_INPUT_PRESENT"

    USER_INPUT_INDEX = [0, 1, 2, 3]

    USER_ACTION = dict()
    USER_ACTION["APPLY_FORCE_LEFT"] = 0
    USER_ACTION["APPLY_FORCE_RIGHT"] = 1
    USER_ACTION["EXIT"] = 2
    USER_ACTION["NO_ACTION"] = 3

    def __init__(self):

        """
        Constructor
        """

        # Initializing the environment
        self.env = CartPoleEnv()
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        # Initializing the neural network
        self.dqn = CartpoleDQN(self.observation_space, self.action_space)

    def getch(self):

        """
        This method gets the user input without requiring the user to press
        enter afterwards
        """

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return self.USER_INPUT[int(ch)]

    def get_user_input(self):

        """Gets the user input and parses the corresponding user action"""

        user_action = self.USER_ACTION["NO_ACTION"]
        user_input = self.getch()

        # Getting user action
        while user_input not in self.USER_INPUT_INDEX:

            print("Please enter an input:")
            user_input = self.getch()
            print("User input: {}".format(user_input))

            user_action = self.USER_ACTION[user_input]

        return user_input , user_action

    def plot_loss(self):

        """
        Plots the loss across the episodes which have been run
        """

        print(self.loss_aggregation)
        plt.plot(self.loss_aggregation)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Step')
        plt.show()

    def run(self):

        """
        Runs the cartpole game (main program entry point)
        """

        # The number of episodes which have completed
        episode = 20

        # The  maximum number of episodes to run
        episode_limit = 40

        # Stores the loss values across all episodes
        self.loss_aggregation = []

        # Initializing the user input
        user_input = self.USER_INPUT[3]

        while user_input != "NO_INPUT_PRESENT" and episode_limit <= 40:

            # Environment reset
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_space])
            step = 0

            # Running the episode
            print('Epsiode: {}'.format(episode))
            while True:

                # Rendering the ste[
                step += 1
                self.env.render()

                # Getting the user action based on the specified mode
                if not self.IMITATION_MODE:
                    user_action = None
                else:
                    user_input, user_action = self.get_user_input()

                # Exiting on user request
                # This will also save the model and plot the loss
                if user_input == "EXIT":
                    print("Saving model...")
                    loss = self.dqn.experience_replay(save=True)
                    self.loss_aggregation.append(loss)
                    self.plot_loss()
                    print("Saved model.")
                    break

                # Getting the machine action
                machine_action = self.dqn.act(state)

                # Printing actions
                print("User Action: {} Machine Action: {}".format(user_action, machine_action))

                # Computing the state
                state_next, reward, terminal, info = self.env.step(machine_action, user_action=user_action)

                # Computing the reward
                reward = reward if not terminal else -reward

                # Computing the next state
                state_next = np.reshape(state_next, [1, self.observation_space])

                # Storing the step for experience replay
                self.dqn.remember(state, machine_action, reward, state_next, terminal)

                # Setting the current state to be the next state
                state = state_next

                # Checking if game over
                if terminal:
                    print("Episode: {} Exploration: {} Score: {}".format(episode, self.dqn.exploration_rate, step))
                    episode += 1
                    break

                # Post processing
                if episode % 20 == 0:
                    print('Saving models...')
                    loss = self.dqn.experience_replay(save=True)
                else:
                    loss = self.dqn.experience_replay(save=False)

                # Adds loss to plot list, if replay buffer is ready for training
                if loss != -1:
                    self.loss_aggregation.append(loss)

                # Getting ready for next state
                print("Reward: {} Step: {} Eposiode :{}".format(reward, step, episode))


if __name__ == "__main__":

    cartpole = Cartpole()
    cartpole.run()
