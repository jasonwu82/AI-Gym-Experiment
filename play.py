import gym
import numpy as np
from collections import defaultdict


step_size = 0.1
discount_factor = 0.5


def truncate_obser(obser):
    return tuple(np.around(obser, decimals=1).tolist())


def train(env):
    util = np.zeros([env.observation_space.n, env.action_space.n])
    for i in range(30000):
        ob = env.reset()
        #ob = truncate_obser(ob)
        for _ in range(50):
            #env.render()
            #act = env.action_space.sample()
            act = np.argmax(util[ob])
            ob_n, r_n, done, info = env.step(act)
            #env.render()
            if done:
                break
            #ob = truncate_obser(ob)
            #ob_n = truncate_obser(ob_n)
            act_n = np.argmax(util[ob_n])
            util[ob, act] = util[ob, act] + step_size*(r_n + discount_factor*util[ob_n, act_n] - util[ob, act])
            ob = ob_n
        run(env, util)

    return util


def run(env, util, render=False):
    ob = env.reset()
    total_r = 0
    for _ in range(50):
        if render:
            env.render()
        act = np.argmax(util[ob])
        ob_n, r_n, done, info = env.step(act)
        total_r += r_n
        if done:
            break
        ob = ob_n
    print("total reward: {0}".format(total_r))
    return total_r


env = gym.make('Taxi-v2')
print(env.action_space)
#env = gym.make('SpaceInvaders-v0')
util = train(env)
for _ in range(10):
    run(env, util, render=True)
