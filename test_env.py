import numpy as np
import gym
import time
import gym.envs.robotics.fetch_env

env = gym.make('FetchPush-v1')
obs = env.reset()
done = False
#
# print(env.action_space)
# action = env.action_space.sample()
# observation, reward, _, info = env.step(action)
# print(observation)
# env.render()
# time.sleep(2)
# action = np.array([1,1,1,1])
# print(action)
# observation, reward, _, info = env.step(action)
# print(observation)
# env.render()

no_action = np.array([0,0,0,0])
action = np.array([1,0,0,1])

while(True):
    action = no_action
    observation, reward, _, info = env.step(action)
    env.render()
    time.sleep(.2)
    print(observation)