import gym
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

#env = gym.make('gym_CarRacing:CarRacing-v1') 
env = DummyVecEnv([lambda: gym.make('gym_CarRacing:CarRacing-v1')])

model = PPO2.load('car_racing_weights.pkl')
model.set_env(env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

#action = env.action_space.sample()
