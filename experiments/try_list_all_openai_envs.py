import gym
from gym import envs

for d in envs.registry.all():
    try:
        env_id = d.id
        print(env_id)
        env = gym.make(env_id)
        print("action space", env.action_space)
        print("observation space", env.observation_space)
        print("observation space shape", env.observation_space.shape)
        if env_id == 'CartPole-v0':
            print('a')
        if env.observation_space.shape is None:
            print('b')
    except:
        print("not successful")
    print()
