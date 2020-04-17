import gym
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
#from gym_CarRacing.envs.CarRacing_env import CarRacingEnv, original_reward_callback

#env = lambda :  CarRacingEnv(
    #grayscale=1,
    #show_info_panel=0,
    #discretize_actions="hard",
    #frames_per_state=4,
    #num_lanes=1,
    #num_tracks=1,
    #reward_fn=original_reward_callback
    #)
#env = DummyVecEnv([env])

env = DummyVecEnv([lambda: gym.make('gym_CarRacing:CarRacing-v1')])

model = PPO2.load('car_racing_weights.pkl')
model.set_env(env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

#action = env.action_space.sample()
