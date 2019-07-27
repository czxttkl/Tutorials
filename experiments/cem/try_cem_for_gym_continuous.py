from typing import Optional, NamedTuple, Tuple
import itertools
import random
import numpy as np
import torch.nn as nn
import logging
import scipy.stats as stats
import gym
import copy


logger = logging.getLogger(__name__)


class StateAction(NamedTuple):
    state: np.ndarray
    action: np.ndarray


class PlanningPolicyInput(NamedTuple):
    state: np.ndarray


class CEMPlannerNetwork(nn.Module):
    def __init__(
        self,
        env,
        cem_num_iterations: int,
        cem_population_size: int,
        num_elites: int,
        plan_horizon_length: int,
        state_dim: int,
        action_dim: int,
        discrete_action: bool,
        gamma: float,
        alpha: float = 0.25,
        epsilon: float = 0.001,
        action_upper_bounds: Optional[np.ndarray] = None,
        action_lower_bounds: Optional[np.ndarray] = None,
    ):
        """
        :param env: open ai gym environment
        :param cem_num_iterations: The maximum number of iterations for searching the best action
        :param cem_population_size: The number of candidate solutions to evaluate in each CEM iteration
        :param num_elites: The number of elites kept to refine solutions in each iteration
        :param plan_horizon_length: The number of steps to plan ahead
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param discrete_action: If actions are discrete or continuous
        :param gamma: The reward discount factor
        :param alpha: The CEM solution update rate
        :param epsilon: The planning will stop early when the solution variance drops below epsilon
        :param action_upper_bounds: Upper bound of each action dimension.
            Only effective when discrete_action=False.
        :param action_lower_bounds: Lower bound of each action dimension.
            Only effective when discrete_action=False.
        """
        super().__init__()
        self.env = env
        self.cem_num_iterations = cem_num_iterations
        self.cem_pop_size = cem_population_size
        self.num_elites = num_elites
        self.plan_horizon_length = plan_horizon_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.discrete_action = discrete_action
        assert self.cem_pop_size > self.num_elites
        if not self.discrete_action:
            assert (
                action_upper_bounds.shape == action_lower_bounds.shape == (action_dim,)
            )
            assert np.all(action_upper_bounds >= action_lower_bounds)
            self.action_upper_bounds = np.tile(
                action_upper_bounds, self.plan_horizon_length
            )
            self.action_lower_bounds = np.tile(
                action_lower_bounds, self.plan_horizon_length
            )

    def forward(self, input: PlanningPolicyInput):
        assert input.state.shape == (self.state_dim, )
        if self.discrete_action:
            return self.discrete_planning(input)
        return self.continuous_planning(input)

    def acc_rewards_of_one_solution(
        self, init_state: np.ndarray, solution: np.ndarray, solution_idx: int
    ):
        """
        one trajectory will be sampled to evaluate a
        CEM solution. Each trajectory is generated by one world model

        :param init_state: its shape is (state_dim, )
        :param solution: its shape is (plan_horizon_length, action_dim)
        :param solution_idx: the index of the solution
        :return reward: Reward of each of trajectories
        """
        reward_vec = np.zeros(self.plan_horizon_length)

        state = init_state
        for j in range(self.plan_horizon_length):
            env_input = StateAction(
                state=state,
                action=solution[j, :],
            )
            reward, next_state, not_terminal = self.sample_reward_next_state_terminal(
                env_input
            )
            reward_vec[j] = reward * (self.gamma ** j)
            if not not_terminal:
                break
            state = next_state

        # CZXTTKL
        # np.set_printoptions(threshold=100000)
        # print(f"{reward_vec}")
        return np.sum(reward_vec)

    def acc_rewards_of_all_solutions(
        self, input: PlanningPolicyInput, solutions: np.ndarray
    ) -> float:
        """
        Calculate accumulated rewards of solutions.

        :param input: the input which contains the starting state
        :param solutions: its shape is (cem_pop_size, plan_horizon_length, action_dim)
        :returns: a vector of size cem_pop_size, which is the reward of each solution
        """
        acc_reward_vec = np.zeros(self.cem_pop_size)
        init_state = input.state
        for i in range(self.cem_pop_size):
            # czxttkl
            discrete_solution = np.argmax(solutions[i], axis=1)
            # print(f"Obtaining {i}-th solution {discrete_solution} reward")
            acc_reward_vec[i] = self.acc_rewards_of_one_solution(
                init_state, solutions[i], i
            )
        return acc_reward_vec

    def sample_reward_next_state_terminal(
        self, env_input: StateAction
    ):
        """ Sample one-step dynamics based on the provided env """
        env = copy.deepcopy(self.env)
        env.env.state = env_input.state
        if self.discrete_action:
            action = np.argmax(env_input.action)
        else:
            action = env_input.action
        next_state, reward, terminal, _ = env.step(action)
        not_terminal = not terminal
        return reward, next_state, not_terminal

    def constrained_variance(self, mean, var):
        lb_dist, ub_dist = (
            mean - self.action_lower_bounds,
            self.action_upper_bounds - mean,
        )
        return np.minimum(np.minimum((lb_dist / 2) ** 2, (ub_dist / 2) ** 2), var)

    def continuous_planning(self, input: PlanningPolicyInput) -> np.ndarray:
        mean = (self.action_upper_bounds + self.action_lower_bounds) / 2
        var = (self.action_upper_bounds - self.action_lower_bounds) ** 2 / 16
        normal_sampler = stats.truncnorm(
            -2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean)
        )

        for i in range(self.cem_num_iterations):
            const_var = self.constrained_variance(mean, var)
            solutions = (
                normal_sampler.rvs(
                    size=[self.cem_pop_size, self.action_dim * self.plan_horizon_length]
                )
                * np.sqrt(const_var)
                + mean
            )
            action_solutions = solutions.reshape(
                (self.cem_pop_size, self.plan_horizon_length, self.action_dim)
            )
            if np.any(action_solutions > 3) or np.any(action_solutions < -3):
                print(action_solutions)
            acc_rewards = self.acc_rewards_of_all_solutions(input, action_solutions)
            elite_mean_reward = np.mean(np.sort(acc_rewards)[-self.num_elites:])
            print(f"{i}-th iteration mean elite reward {elite_mean_reward}")
            elites = solutions[np.argsort(acc_rewards)][-self.num_elites:]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            print(f"var={var}, max={np.max(var)}")
            if np.max(var) <= self.epsilon:
                print(f"break because var")
                break

        # Pick the first action of the optimal solution
        solution = mean[: self.action_dim]
        return solution

    def discrete_planning(
        self, input: PlanningPolicyInput
    ) -> Tuple[int, np.ndarray]:
        # For discrete actions, we use random shoots to get the best next action
        random_action_seqs = list(
            itertools.product(range(self.action_dim), repeat=self.plan_horizon_length)
        )
        random_action_seqs = random.choices(random_action_seqs, k=self.cem_pop_size)
        action_solutions = np.zeros(
            (self.cem_pop_size, self.plan_horizon_length, self.action_dim)
        )
        for i, action_seq in enumerate(random_action_seqs):
            for j, act_idx in enumerate(action_seq):
                action_solutions[i, j, act_idx] = 1
        acc_rewards = self.acc_rewards_of_all_solutions(input, action_solutions)

        first_action_tally = np.zeros(self.action_dim)
        reward_tally = np.zeros(self.action_dim)
        for action_seq, acc_reward in zip(random_action_seqs, acc_rewards):
            first_action = action_seq[0]
            first_action_tally[first_action] += 1
            reward_tally[first_action] += acc_reward

        best_next_action_idx = np.nanargmax(reward_tally / first_action_tally)
        best_next_action_one_hot = np.zeros(self.action_dim)
        best_next_action_one_hot[best_next_action_idx] = 1

        # czxttkl
        print(
            f"choose action {reward_tally} / {first_action_tally} = {reward_tally/first_action_tally} best_next_action: {best_next_action_idx}"
        )
        return best_next_action_idx, best_next_action_one_hot


def try_cem(env, test_episodes, cem_pop_size, plan_horizon, num_elites, cem_num_iterations):
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action = True
        action_dim = env.action_space.n
    else:
        discrete_action = False
        action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    cem_planner_network = CEMPlannerNetwork(
        env,
        cem_num_iterations=cem_num_iterations,
        cem_population_size=cem_pop_size,
        num_elites=num_elites,
        plan_horizon_length=plan_horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_action=discrete_action,
        gamma=1.0,
        alpha=0.25,
        epsilon=0.001,
        action_upper_bounds=env.action_space.high,
        action_lower_bounds=env.action_space.low,
    )

    all_episodes_rewards = []
    for i_episode in range(test_episodes):
        observation = env.reset()
        episode_rewards = []
        for t in range(1000):
            print(f"\nRun {i_episode}-th episode {t}-th step  {observation}")
            action = cem_planner_network(PlanningPolicyInput(state=observation))
            observation, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            print(f"reward: {reward}, acc: {np.sum(episode_rewards)}\n")
            if done:
                print("Episode finished after with {} reward".format(np.sum(episode_rewards)))
                all_episodes_rewards.append(np.sum(episode_rewards))
                break

    with open("try_cem_for_gym.txt", "a") as f:
        write_str = f"TEST_EPISODES={test_episodes}, CEM_POP_SIZE={cem_pop_size}, NUM_ELITES={num_elites}, PLAN_HORIZON={plan_horizon}, CEM_ITERATIONS={cem_num_iterations}" \
                    f", REWARD={np.mean(all_episodes_rewards)}\n"
        print(write_str)
        f.write(write_str)


if __name__ == "__main__":
    # # Test env state can be recovered
    # env = gym.make("MountainCarContinuous-v0")
    # env.seed(0)

    MAX_STEPS = 4
    STATE_DIM = 3
    ACTION_DIM = 2
    from experiments.cem.lqr_env import Env
    env = Env(MAX_STEPS, STATE_DIM, ACTION_DIM)

    cur_state = env.reset()
    action = np.array([0.1, 0.1])
    print("cur_state", cur_state)
    print("env.env.state", env.env.state)
    next_state, reward, done, info = env.step(action)
    print("next state", next_state)
    next_next_state, reward, done, info = env.step(action)
    print("next next state", next_next_state)

    env.env.state = next_state
    recovered_next_next_state, reward, done, info = env.step(action)
    print("recovered next next state", recovered_next_next_state)
    assert np.all(next_next_state == recovered_next_next_state)

    env.env.state = cur_state
    recovered_next_state, reward, done, info = env.step(action)
    print("recovered next state", recovered_next_state)
    assert np.all(next_state == recovered_next_state)

    TEST_EPISODES = 1
    CEM_POP_SIZE = 500
    PLAN_HORIZON = 4
    CEM_NUM_ITERATIONS = 150
    NUM_ELITES = 50
    try_cem(env, TEST_EPISODES, CEM_POP_SIZE, PLAN_HORIZON, NUM_ELITES, CEM_NUM_ITERATIONS)










