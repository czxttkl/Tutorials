from model.lstm import LSTM
from model.dqn import DQN


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def get_env(env_str, device):
    if env_str == 'finite':
        from env.finite_gridworld_env import GridWorldEnv
        env = GridWorldEnv(device)
    elif env_str == 'rnn':
        from env.rnn_gridworld_env import GridWorldEnv
        env = GridWorldEnv(device)
    elif env_str == 'lunar':
        from env.lunar_env import LunarEnv
        env = LunarEnv(device)
    elif env_str == 'cartpole':
        from env.cartpole_env import CartPoleEnv
        env = CartPoleEnv(device)
    return env


def get_model(model_str, env, gamma, replay_memory_size, batch_size, device):
    if model_str == 'lstm':
        lstm_input_dim = lstm_output_dim = env.action_dim
        lstm_hidden_dim = 50
        lstm_num_layer = 2
        policy_net = LSTM(lstm_input_dim, lstm_num_layer,
                          lstm_hidden_dim, lstm_output_dim,
                          gamma, replay_memory_size, batch_size).to(device)
        target_net = None
    elif model_str == 'dqn':
        dqn_input_dim = env.state_dim
        dqn_hidden_dim = 64
        dqn_num_layer = 2
        dqn_output_dim = env.action_dim
        parametric = True
        policy_net = DQN(dqn_input_dim, dqn_num_layer,
                         dqn_hidden_dim, dqn_output_dim,
                         parametric, gamma, replay_memory_size,
                         batch_size).to(device)
        # target_net = None
        target_net = DQN(dqn_input_dim, dqn_num_layer,
                         dqn_hidden_dim, dqn_output_dim,
                         parametric, gamma, replay_memory_size,
                         batch_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    else:
        raise Exception

    return policy_net, target_net


def adjust_epsilon(cur_episode, final_episode, learning_start_episodes, epsilon_thres):
    if cur_episode < learning_start_episodes:
        # make policy network select random actions
        return 1
    temp_epsilon = (0.995 ** (cur_episode - learning_start_episodes)) * 1
    if temp_epsilon < epsilon_thres:
        return epsilon_thres
    else:
        return temp_epsilon
