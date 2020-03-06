import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('gym_gridworld:gridworld-v0')
env.render() # not required, just for visualisation purposes

#function for finding the action with the most value when following greedy policy
def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions]) # look at value estimates for each state action pair
    maxActionIndex = np.argmax(values) # which value is the highest
    #return actions[maxActionIndex] # and which action corresponds to that highest value
    return maxActionIndex # decided to return index instead since changed step() in gridworld_env.py to work with index as that's what gym's sample() returns

# model hyperparameters
ALPHA = 0.1 # learning rate
GAMMA = 1.0 # discount factor, so not using discounting here
EPS = 1.0 # value which will be decayed over training and will determine whether act greedily or randomly

# initialise Q function - sometimes will initialise with random values but here
# we are setting all values to zero to encourage exploration (impossible to get
# return of zero or above since -1 per timestep so for every action consideration
# agent will observe a return worse than zero and want to try other options)
Q = {}
for state in env.stateSpacePlus:
    for action in env.possibleActions:
        Q[state,action] = 0

numGames = 1000 # lowered from his example 50000 because has learnt optimal policy well before 1000 games even 
totalRewards = np.zeros(numGames) # array where store individual episode rewards

for i in range(numGames):
    #sanity check
    if i % 5000 == 0:
        print('starting game', i)

    # at the start of every episode reset everything
    done = False
    epRewards = 0
    observation = env.reset()

    # main training loop until episode ends
    while not done:
        rand = np.random.random()
        # first line is acting greedily, second is obviously random (both returning an index 0-3)
        action = maxAction(Q, observation, env.possibleActions) if rand < (1-EPS) \
                else env.action_space.sample()
        observation_, reward, done, info = env.step(action) # take action, works because action = 0-3
        epRewards += reward # cumulate reward

        #regardless of whether acted greedily or randomly, the greedy action is set as at+1 for purpose of update
        action_ = maxAction(Q, observation_, env.possibleActions)
        # update Q with TD update rule
        # the 'action' used in this line is being used as a dictionary key so needs to turn back from 0-3 to U/R/L/D
        # hence why action is now env.possibleActions[action]
        Q[observation, env.possibleActions[action]] = Q[observation, env.possibleActions[action]] + ALPHA*(reward + \
                GAMMA*Q[observation_, env.possibleActions[action_]] - Q[observation, env.possibleActions[action_]])

        # update environment
        observation = observation_

    # decay epsilon - would typically use log or sqrt or something but here just decaying linearly
    if EPS - 2 / numGames > 0:
        EPS = 2 / numGames
    else:
        EPS = 0

    totalRewards[i] = epRewards # log episode reward

plt.plot(totalRewards)
plt.show()
