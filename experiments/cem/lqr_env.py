"""
Simple linear dynamic system
https://www.argmin.net/2018/02/08/lqr/
"""
import scipy.linalg as linalg
import numpy as np
from gym.spaces import Box, Discrete


def lqr(A, B, Q, R, state):
    # Solves for the optimal infinite-horizon LQR gain matrix given linear system (A,B)
    # and cost function parameterized by (Q,R)

    # solve DARE:
    M = linalg.solve_discrete_are(A, B, Q, R)

    # K=(B'MB + R)^(-1)*(B'MA)
    K = np.dot(linalg.inv(np.dot(np.dot(B.T, M), B) + R), (np.dot(np.dot(B.T, M), A)))

    state = state.reshape((-1, 1))
    action = -K.dot(state)
    return action.squeeze()


class Env:
    def __init__(self, max_steps, state_dim, action_dim):
        self.env = LinDynaEnv(max_steps, state_dim, action_dim)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def seed(self, s):
        pass

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class LinDynaEnv:
    """
    A linear dynamical system characterized by A, B, Q, and R.

    Suppose x_t is current state, u_t is current action, then:

    x_t+1 = A x_t + B u_t
    Reward_t = x_t' Q x_t + u_t' R u_t
    """

    def __init__(self, max_steps, state_dim, action_dim):
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = Box(low=-3, high=3, shape=(action_dim,))
        self.observation_space = Box(low=-1.7976931348623157e+308, high=1.7976931348623157e+308, shape=(state_dim,))

        valid_env = False
        while not valid_env:
            self.A, self.B, self.S, self.Q, self.R = self.gen_mats()
            # this matrix should be positive definite
            check_mat = np.vstack((np.hstack((self.Q, self.S)), np.hstack((self.S.T, self.R))))
            valid_env = self.is_pos_def(check_mat)
            print("Re-initialize environment")

        print(f"A:\n{self.A}\nB:\n{self.B}\nQ:{self.Q}\nR:{self.R}\n")

    def gen_mats(self):
        A = np.random.randint(low=-2, high=3, size=(self.state_dim, self.state_dim))
        B = np.random.randint(low=-2, high=3, size=(self.state_dim, self.action_dim))
        S = np.zeros((self.state_dim, self.action_dim))
        # Q and R should be symmetric
        Q = np.random.randint(low=-2, high=3, size=(self.state_dim, self.state_dim))
        Q = (Q + Q.T) / 2
        R = np.random.randint(low=-2, high=3, size=(self.action_dim, self.action_dim))
        R = (R + R.T) / 2
        return A, B, S, Q, R

    @staticmethod
    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    def reset(self):
        self.state = np.random.randint(low=-1, high=2, size=(self.state_dim,))
        self.step_cnt = 0
        return self.state

    def step(self, action):
        assert len(action) == self.action_dim
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.reshape((self.action_dim, 1))
        state = self.state.reshape((self.state_dim, 1))
        next_state = self.A.dot(state) + self.B.dot(action)
        next_state = next_state.squeeze()
        # add the negative sign because we actually want to maximize the rewards, while an LRQ solution minimizes
        # rewards by convention
        reward = - (state.T.dot(self.Q).dot(state) + action.T.dot(self.R).dot(action))
        reward = reward.squeeze()
        self.step_cnt += 1
        terminal = False
        if self.step_cnt >= self.max_steps:
            terminal = True
        self.state = next_state
        return next_state, reward, terminal, None


if __name__ == "__main__":
    NUM_EPISODES = 300
    MAX_STEPS = 4
    STATE_DIM = 30
    ACTION_DIM = 2
    np.random.seed(7)
    env = Env(MAX_STEPS, STATE_DIM, ACTION_DIM)

    # random action
    all_episode_reward = []
    for i in range(NUM_EPISODES):
        episode_reward = []
        cur_state = env.reset()
        for t in range(MAX_STEPS):
            action = np.random.uniform(env.action_space.low, env.action_space.high, ACTION_DIM)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, reward, terminal, _ = env.step(action)
            episode_reward.append(reward)
            print(f"Step {t}, obs {cur_state}, action {action}, reward {reward}, terminal {terminal}")
            cur_state = next_state
        print(f"episode {i} accumulated reward: {np.mean(episode_reward)}\n")
        all_episode_reward.append(np.mean(episode_reward))


    # LQR
    all_episode_reward_lqr = []
    for i in range(NUM_EPISODES):
        episode_reward = []
        cur_state = env.reset()
        for t in range(MAX_STEPS):
            action = lqr(env.env.A, env.env.B, env.env.Q, env.env.R, cur_state)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, reward, terminal, _ = env.step(action)
            episode_reward.append(reward)
            print(f"Step {t}, obs {cur_state}, action {action}, reward {reward}, terminal {terminal}")
            cur_state = next_state
        print(f"episode {i} accumulated reward: {np.mean(episode_reward)}\n")
        all_episode_reward_lqr.append(np.mean(episode_reward))

    print(f"Random policy: average episode accumulated reward: {np.mean(all_episode_reward)}\n\n")
    print(f"LQR policy: average episode accumulated reward: {np.mean(all_episode_reward_lqr)}")
