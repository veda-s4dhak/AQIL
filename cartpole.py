"""
This is the main application entry point. It runs the cartpole game
via the cartpole_dqn and cartpole_env modules.
"""

import numpy as np
import sys, termios, tty, os
from cartpole_env import CartPoleEnv
from cartpole_dqn import CartpoleDQN
import matplotlib.pyplot as plt
import pandas as pd
import json


class Cartpole:
    """
    Cartpole runs the game using the deep neural network and the OpenAI Gym
    """

    USER_ACTION = dict()
    USER_ACTION[1] = "APPLY FORCE RIGHT"
    USER_ACTION[2] = "APPLY FORCE LEFT"
    USER_ACTION[0] = "EXIT"

    USER_INPUT_INDEX = [0, 1, 2]

    def __init__(self, **kwargs):

        """
        Constructor
        """

        self.config = kwargs

        self.threadID = kwargs['model_name']
        self.name = kwargs['model_name']

        # Setting configuration
        self.USER_IMITATION_MODE = kwargs['user_imitation_mode']
        self.PID_IMITATION_MODE = kwargs['pid_imitation_mode']

        # Initializing the environment
        self.env = CartPoleEnv()
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        # Initializing the neural network
        self.model_name = kwargs['model_name']

        # The maximum number of episodes to run
        self.n_episodes = kwargs['n_episodes']

        # Initializing the model
        self.dqn_params = kwargs
        self.dqn_params['observation_space'] = self.observation_space
        self.dqn_params['action_space'] = self.action_space
        self.dqn = CartpoleDQN(**self.dqn_params)

        # Average training loss per step
        self.loss_aggregation = []

        # Total Reward per Episode
        self.reward_aggregation = []

        # User Action per Step
        self.user_action_aggregation = []

        # Machine Action per Step
        self.machine_action_aggregation = []
        self.score_aggregation = []

        # Creates directory if directory does not exist
        if not os.path.exists('.//models'):
            os.mkdir('.//models')

        if not os.path.exists('.//plots'):
            os.mkdir('.//plots')

        # Required for PID Control
        self.P = 0
        self.I = 0
        self.D = 0
        self.prev_error = 0

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

    def get_user_action(self):

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

    def get_pid_action(self):

        # PID Constants
        kP = self.config["P"] # 0.3 Optimal
        kI = self.config["I"] # 0.1 Optimal
        kD = self.config["D"] # 10 Optimal
        desired_angle = 0

        # 1) Get the pole angle
        pole_angle = self.env.theta

        # Error computation
        error = desired_angle - pole_angle

        # 2) Compute action
        self.P = error
        self.I += error
        self.D = error - self.prev_error
        action = kP * self.P + kI * self.I + kD * self.D

        self.prev_error = error

        return (1 if action < 0
                else 0)

    def plot_data(self):

        """
        Plots the loss across the episodes which have been run
        """

        figure_title = '{} Experiment Results'.format(self.model_name)

        fig = plt.figure(figure_title, figsize=(8, 15))
        nrows = 4 if self.USER_IMITATION_MODE else 3

        # Plots Model Loss graph
        ax1 = fig.add_subplot(nrows, 1, 1)
        plt.plot(self.loss_aggregation)
        ax1.set_yscale('log')
        plt.title('Model Loss')
        plt.ylabel('Average Loss')
        plt.xlabel('Step')

        # Plots Reward graph
        ax2 = plt.subplot(nrows, 1, 2)
        plt.plot(self.reward_aggregation)
        plt.title('Reward')
        plt.ylabel('Reward')
        plt.xlabel('Episode')

        # Plots Machine Action Graph
        ax3 = plt.subplot(nrows, 1, 3)
        plt.plot(self.machine_action_aggregation)
        plt.title('Machine Action')
        plt.ylabel('Action')
        plt.xlabel('Step')

        # Plots User Action Graph if IMITATION_MODE
        if self.USER_IMITATION_MODE:
            ax4 = plt.subplot(nrows, 1, 4)
            plt.plot(self.user_action_aggregation)
            plt.title('User Action')
            plt.ylabel('Action')
            plt.xlabel('Step')

        plt.tight_layout(pad=3, h_pad=3)

        plt.savefig(os.path.join(".", "plots", "{}.png".format(self.model_name)), bbox_inches='tight')
        plt.show()

        with open(os.path.join(".", "plots", "{}.txt".format(self.model_name)), 'w') as f:
            json.dump(self.config, f)

        # Generates dicts for saving to csv
        loss_aggregation_dict = dict()
        action_dict = dict()
        reward_dict = dict()

        # Creates dict for loss data
        loss_aggregation_dict['Loss'] = self.loss_aggregation

        # Creates dict for User Action data if IMITATION_MODE
        if self.USER_IMITATION_MODE:
            action_dict['User_Action'] = self.user_action_aggregation

        # Creates dict for Machine Action data
        action_dict['Machine_Action'] = self.machine_action_aggregation

        # Creates dict for Reward data
        reward_dict['Reward'] = self.reward_aggregation
        # reward_dict['Reward'] = self.reward_aggregation

        # for episode_num in range(0, len(self.loss_aggregation)):
        #     loss_aggregation_dict[episode_num] = self.loss_aggregation[episode_num]

        # Saving the data to a csv
        df = pd.DataFrame.from_dict(loss_aggregation_dict)
        df.to_csv(os.path.join(".", "plots", "{}.csv".format(self.model_name + '_loss')), header=True, index=True)

        df = pd.DataFrame.from_dict(action_dict)
        df.to_csv(os.path.join(".", "plots", "{}.csv".format(self.model_name + '_action')), header=True, index=True)

        df = pd.DataFrame.from_dict(reward_dict)
        df.to_csv(os.path.join(".", "plots", "{}.csv".format(self.model_name + '_reward')), header=True, index=True)

    def run(self):

        """
        Runs the cartpole game (main program entry point)
        """
        # The number of episodes which have completed
        episode = 0

        user_action_string = None

        while (user_action_string != "EXIT") and (episode <= self.n_episodes):

            # Environment reset
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_space])
            step = 0

            # Episode Reward
            r_episode = 0

            # Running the episode
            print('Episode: {}'.format(episode))
            while True:

                # Rendering the step
                step += 1
                self.env.render()

                # Getting the user action based on the specified mode
                if not self.USER_IMITATION_MODE:
                    user_action_string = None
                    user_action = None
                else:
                    user_action, user_action_string = self.get_user_action()
                    user_action -= 1
                    self.user_action_aggregation.append(user_action)

                if self.USER_IMITATION_MODE:
                    user_action, user_action_string = self.get_user_action()
                    user_action -= 1
                    self.user_action_aggregation.append(user_action)
                elif self.PID_IMITATION_MODE:
                    pid_action = self.get_pid_action()
                    self.user_action_aggregation.append(pid_action)
                    user_action = pid_action
                else:
                    user_action_string = None
                    user_action = None

                # Exiting on user request
                # This will also save the model and plot the loss
                if user_action_string == "EXIT":
                    print("Saving model...")
                    loss, r = self.dqn.experience_replay(save=True)
                    self.loss_aggregation.append(loss)
                    self.reward_aggregation.append(r_episode)
                    print("Saved model.")
                    break

                # Getting the machine action
                machine_action = self.dqn.act(state)

                # Records machine action for step step
                self.machine_action_aggregation.append(machine_action)

                # Printing actions
                if self.USER_IMITATION_MODE or self.PID_IMITATION_MODE:
                    print("User Action: {} Machine Action: {}".format(user_action, machine_action))
                else:
                    print("Machine Action: {}".format(machine_action))

                # Computing the state
                state_next, reward, terminal, info = self.env.step(machine_action, user_input=user_action)

                # Computing the reward
                reward = reward if not terminal else -reward
                r_episode += reward

                state_next = np.reshape(state_next, [1, self.observation_space])

                # Storing the step for experience replay
                self.dqn.remember(state, machine_action, reward, state_next, terminal)

                # Setting the current state to be the next state
                state = state_next

                # Post processing
                loss = 1
                if (episode % 1 == 0) and terminal:
                    print('Saving models...')
                    loss, r_step = self.dqn.experience_replay(save=True)
                else:
                    loss, r_step = self.dqn.experience_replay(save=False)

                # Checking if game over
                if terminal:
                    print("Episode: {} Exploration: {} Score: {}".format(episode, self.dqn.exploration_rate, step))
                    self.reward_aggregation.append(r_episode)
                    self.score_aggregation.append(step)
                    episode += 1

                    # input() # Debugging at the end of every episode
                    break

                # Adds loss to plot list, if replay buffer is ready for training
                if loss != -1:
                    self.loss_aggregation.append(loss)

                # Getting ready for next state
                print("Reward: {} Step: {} Episode: {} Loss: {}".format(reward, step, episode, loss))

        self.plot_data()


def run_cartpole(cfg):
    cp = Cartpole(**cfg)
    cp.run()


if __name__ == "__main__":
    # config = dict()
    # config['model_name'] = "RL200"
    # config['n_episodes'] = 300
    # config['user_imitation_mode'] = False
    # config['pid_imitation_mode'] = True
    #
    # config['gamma'] = 0.95
    # config['learning_rate'] = 1e-5
    # config['exploration_max'] = 1.0
    # config['exploration_min'] = 0.01
    # config['exploration_decay'] = 0.995
    # config['exploration_power'] = 1.005
    # config['exploration_rate'] = 1.00
    #
    # cartpole = Cartpole(**config)
    # cartpole.run()

    config = dict()
    config['model_name'] = "IL250"
    config['n_episodes'] = 250
    config['user_imitation_mode'] = False
    config['pid_imitation_mode'] = True

    config["P"] = 0.6
    config["I"] = 0.00625
    config["D"] = 0.8

    config['gamma'] = 0.95
    config['learning_rate'] = 1e-5
    config['exploration_max'] = 0.9
    config['exploration_min'] = 0.01
    config['exploration_decay'] = 0.995
    config['exploration_power'] = 1.005
    config['exploration_rate'] = 1.00

    # x = threading.Thread(target=run_cartpole, args=(config,))
    # x.start()

    run_cartpole(config)
    # cartpole = Cartpole(**config)
    # cartpole.run()

    # config = dict()
    # config['model_name'] = "RL200"
    # config['n_episodes'] = 300
    # config['user_imitation_mode'] = False
    # config['pid_imitation_mode'] = True
    #
    # config['gamma'] = 0.95
    # config['learning_rate'] = 1e-5
    # config['exploration_max'] = 1.0
    # config['exploration_min'] = 0.01
    # config['exploration_decay'] = 0.995
    # config['exploration_power'] = 1.005
    # config['exploration_rate'] = 1.00
    #
    # cartpole = Cartpole(**config)
    # cartpole.run()
