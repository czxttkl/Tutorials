import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from try_openai_leaderboard_solution import DQNSolver


def test_solution_cartpole():
    agent = DQNSolver(
        env_name='CartPole-v0',
        n_episodes=1000,
        n_win_ticks=195,
        max_env_steps=None,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.2,
        epsilon_log_decay=0.995,
        alpha=0.001,
        alpha_decay=0.0,
        batch_size=256,
        double_q=True,
        loss='mse',
        monitor=False,
    )
    e = agent.run()
    assert e < 1000


def test_solution_lunarlander():
    agent = DQNSolver(
        env_name='LunarLander-v2',
        n_episodes=8000,
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
    )
    e = agent.run()
    assert e < 8000


if __name__ == "__main__":
    # test_solution_cartpole()
    test_solution_lunarlander()
