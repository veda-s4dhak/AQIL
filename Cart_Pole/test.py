import numpy as np
import gym

env = gym.make("CartPole-v1")
observation = env.reset()

# 1. Create Q-Table Structure
Q = np.zeros([env.observation_space.shape[0],env.action_space.n])

# 2. Parameters of Q-leanring
learning_rate = .628
discount_factor = .9
episodes = 5000
rev_list = [] # rewards per episode calculate

for i in range(episodes):

  state = env.reset() # Initializing the environment to the beginning stage
  r_sum = 0 # Summing across all rewards?
  d = False # Game can end when d = True
  j = 0 # Maximum number of steps taken in the episode

  while j < 99:
    env.render()
    j+=1
    import random
    import gym
    import numpy as np
    from collections import deque
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam


    ENV_NAME = "CartPole-v1"

    GAMMA = 0.95
    LEARNING_RATE = 0.001

    MEMORY_SIZE = 1000000
    BATCH_SIZE = 20

    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995


    class DQNSolver:

      def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(
          Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

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
            q_update = (reward + GAMMA * np.amax(
              self.model.predict(state_next)[0]))
          q_values = self.model.predict(state)
          q_values[0][action] = q_update
          self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


    def cartpole():
      env = gym.make(ENV_NAME)
      observation_space = env.observation_space.shape[0]
      action_space = env.action_space.n
      dqn_solver = DQNSolver(observation_space, action_space)
      run = 0
      while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
          step += 1
          # env.render()
          action = dqn_solver.act(state)
          state_next, reward, terminal, info = env.step(action)
          reward = reward if not terminal else -reward
          state_next = np.reshape(state_next, [1, observation_space])
          dqn_solver.remember(state, action, reward, state_next, terminal)
          state = state_next
          if terminal:
            print
            "Run: " + str(run) + ", exploration: " + str(
              dqn_solver.exploration_rate) + ", score: " + str(step)
            break
          dqn_solver.experience_replay()


    if __name__ == "__main__":
      cartpole()
    # Choosing an action
    print(state)
    pertrubation = np.random.randn(1, env.action_space.n) * (1. / (i + 1))
    a = np.argmax(Q[state, :] + pertrubation) # TODO: WHAT IS GOING ON HERE?

    # Get new state & reward from environment
    s1, r, d, _ = env.step(a) # Executing action within the state

    # Update Q-Table with new knowledge
    current_q_value = Q[state, a]
    next_q_value = discount_factor * np.max(Q[s1, :])
    Q[state, a] = Q[state, a] + learning_rate * (r + next_q_value - current_q_value)
    r_sum += r
    s = s1
    if d == True:
      break

  rev_list.append(r_sum)
  env.render()

  print("Reward Sum on all episodes " + str(sum(rev_list) / episodes))
  print("Final Values Q-Table")
  print(Q)
  rev_list.append(r_sum)
  env.render()
