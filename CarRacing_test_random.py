import gym
env = gym.make('CarRacing-v0')
observation = env.reset()
for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        if done:
            print("Finished after {} timesteps".format(t+1))
            break
