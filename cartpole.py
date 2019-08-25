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

    USER_ACTION = dict()
    USER_ACTION[1] = "APPLY FORCE LEFT"
    USER_ACTION[2] = "APPLY FORCE RIGHT"
    USER_ACTION[0] = "EXIT"

    USER_INPUT_INDEX = [0, 1, 2]

    def __init__(self):

        """
        Constructor
        """

        # Initializing the environment
        self.env = CartPoleEnv()
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        # Initializing the neural network
        self.dqn = CartpoleDQN(self.observation_space, self.action_space, model_name='Cartpole_DQN')

        # Stores the loss values across all episodes
        self.loss_aggregation = []

    @staticmethod
    def getch():

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

        return int(ch)

    def get_user_input(self):

        """Gets the user input and parses the corresponding user action"""

        user_action = None

        print("Please enter an input:")
        user_input = self.getch()

        # Getting user action
        while user_input not in self.USER_INPUT_INDEX:
            print("Please enter an input:")
            user_input = int(self.getch())

            print("User input: {}".format(user_input))

        user_action = self.USER_ACTION[user_input]

        return user_input, user_action

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
        episode = 0

        # The  maximum number of episodes to run
        episode_limit = 40

        user_action = None

        while user_action != "EXIT" and episode <= episode_limit:

            # Environment reset
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_space])
            step = 0

            # Running the episode
            print('Episode: {}'.format(episode))
            while True:

                # Rendering the ste[
                step += 1
                self.env.render()

                # Getting the user action based on the specified mode
                if not self.IMITATION_MODE:
                    user_action = None
                    user_input = None
                else:
                    user_input, user_action = self.get_user_input()

                print(user_action)

                # Exiting on user request
                # This will also save the model and plot the loss

                if user_action == "EXIT":
                    print("Saving model...")
                    loss = self.dqn.experience_replay(save=True)
                    self.loss_aggregation.append(loss)
                    print("Saved model.")
                    break

                # Getting the machine action
                machine_action = self.dqn.act(state)

                # Printing actions
                if self.IMITATION_MODE:
                    print("User Action: {} Machine Action: {}".format(user_input, machine_action))
                else:
                    print("Machine Action: {}".format(machine_action))

                # Computing the state
                state_next, reward, terminal, info = self.env.step(machine_action, user_input=user_input)

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
                print("Reward: {} Step: {} Episode :{}".format(reward, step, episode))

        self.plot_loss()


if __name__ == "__main__":
    cartpole = Cartpole()
    cartpole.run()
