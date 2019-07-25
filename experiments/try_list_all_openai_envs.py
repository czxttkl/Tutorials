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
        if isinstance(env.action_space,  gym.spaces.Box) \
                and isinstance(env.observation_space,  gym.spaces.Box)\
                and len(env.observation_space.shape) == 1:
            print('-----good-----')
            print(env.action_space.sample())
            print(env.observation_space.sample())
            print('-------------')
    except Exception as e:
        print("not successful", e)
    print()
