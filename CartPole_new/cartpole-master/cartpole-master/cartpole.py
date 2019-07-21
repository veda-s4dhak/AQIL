import random
import gym
import numpy as np
import tensorflow as tf
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys, termios, tty, os, time
from CartPole_v1 import CartPoleEnv
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras
from datetime import datetime


# from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

class DQNSolver:

    # q_values is the output of the neural network given a state also they
    # are also the labels during training in experience replay

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.logdir = os.path.join('.', "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)

        if not os.path.exists(os.path.join(".", "model.h5")):
            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(24, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        else:
            print('Loading model...')
            self.model = load_model(os.path.join(".", "model.h5"))

        self.save_path = os.path.join(".", "model.h5")
        self.checkpoint = ModelCheckpoint(self.save_path, monitor='loss', verbose=1, save_best_only=False, mode='min')
        self.callbacks_list = [self.checkpoint, self.tensorboard_callback]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0])) # Bellman

            # Output of the neural network (q values) given the state
            q_values = self.model.predict(state)

            # Action is the action which was taken in the state within the episde
            # This action is/was thought to be the optmail action before training
            # This action gets updated with the new reward.
            print("Predicted q: {}".format(q_values))
            q_values[0][action] = q_update
            print("Target q: {}".format(q_values))

            # Backpropagation based on the current experience
            # Note: at this point your q_value are your labels
            self.model.fit(state, q_values, verbose=0, callbacks=self.callbacks_list)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    # env = gym.make(ENV_NAME)
    env = CartPoleEnv()
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    print("Action Space Length: {}".format(action_space))
    print("Action Space: {}".format(env.action_space))
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0



    while True:

        # Environment reset
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0

        # User input init to invalid value
        user_input = -1
        user_action = 0
        while True:

            # Rendering the ste[
            step += 1
            env.render()

            # Getting user action
            while (user_input != 1) and (user_input != 2):

                print("please enter an input")
                # try:
                user_input = int(getch())
                print("user_input: {}".format(user_input))

                if user_input == 1:
                    user_action = 1
                elif user_input == 2:
                    user_action = 0
                # except:
                    # pass

            # Getting the machine action
            machine_action = dqn_solver.act(state)
            env.machine_action = machine_action

            # Setting the action to machine or use
            action = machine_action#user_action

            # Printing actions
            print("User Action: {} Machine Action: {} Action: {}".format(user_action, machine_action, action))

            # Computing the state
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])

            dqn_solver.remember(state, machine_action, reward, state_next, terminal)
            state = state_next

            # Checking if game over
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                #score_logger.add_score(step, run)
                break

            # Post processing
            dqn_solver.experience_replay()

            # Getting ready for next state
            print("Reward: {}".format(reward))

            user_input = -1


if __name__ == "__main__":
    cartpole()
