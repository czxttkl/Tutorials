import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import clone_model
import tensorflow as tf


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


loss_funcs = {'mse': 'mse', 'huber': huber_loss}


class DQNSolver:
    def __init__(
            self,
            env_name,
            n_episodes=20000,
            n_win_ticks=195,
            max_env_steps=None,
            gamma=1.0,
            epsilon=1.0,
            epsilon_min=0.2,
            epsilon_log_decay=0.995,
            alpha=0.001,
            alpha_decay=0.0,
            batch_size=256,
            double_q=True,
            loss='mse',
            monitor=False,
    ):
        self.memory = deque(maxlen=200000)
        self.env = gym.make(env_name)
        if monitor:
            self.env = gym.wrappers.Monitor(self.env, '../data/{}'.format(env_name), force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.double_q = double_q
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=self.env.observation_space.shape[0], activation='tanh'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.env.action_space.n, activation='linear'))
        self.model.compile(loss=loss_funcs[loss], optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        self.target_model = clone_model(self.model)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
            self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.env.observation_space.shape[0]])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            if self.double_q:
                y_target = self.target_model.predict(state)
            else:
                y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        res = self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return res.history['loss'][0]

    def run(self):
        final_scores = deque(maxlen=100)
        accu_scores = deque(maxlen=100)
        losses = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            accu_reward = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                accu_reward += reward

            final_scores.append(reward)
            accu_scores.append(accu_reward)
            mean_final_score = np.mean(final_scores)
            mean_accu_score = np.mean(accu_scores)
            if accu_reward >= self.n_win_ticks:
                print('Solved after {} trials, score {} ✔'.format(e, accu_reward))
                return e - 100

            if e % 10 == 0:
                self.target_model.set_weights(self.model.get_weights())

            loss = self.replay(self.batch_size)
            losses.append(loss)
            mean_loss = np.mean(losses)

            if e % 100 == 0:
                print('[Episode {}] - Last 100 episodes final reward: {}, accu reward: {}, mean loss: {}'
                      .format(e, mean_final_score, mean_accu_score, mean_loss))

        print('Did not solve after {} episodes 😞'.format(e))
        return 9999999999


if __name__ == '__main__':
    agent = DQNSolver(env_name='CartPole-v0')
    # agent = DQNSolver(env_name='LunarLander-v2')
    agent.run()
