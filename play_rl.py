import gym
import numpy as np
from collections import defaultdict
import time


step_size = 0.1
discount_factor = 0.99
eligible_factor = 0.7


def truncate_obser(obser):
    return tuple(np.around(obser, decimals=1).tolist())


def td_zero(util, ob, act, delta):
    util[ob, act] = util[ob, act] + step_size * delta
    return util


def td_lambda(util, ob, act, delta, eligible_mat, eligible_factor):
    eligible_mat = discount_factor*eligible_factor*eligible_mat
    eligible_mat[ob, act] += 1.0
    util += step_size * delta * eligible_mat
    return util, eligible_mat


def train(env):
    util = np.zeros([env.observation_space.n, env.action_space.n])
    eligible_mat = np.zeros([env.observation_space.n, env.action_space.n])
    for i in range(30000):
        ob = env.reset()
        for _ in range(50):

            act = np.argmax(util[ob])
            ob_n, r_n, done, info = env.step(act)

            if done:
                break

            act_n = np.argmax(util[ob_n])
            delta = r_n + discount_factor*util[ob_n, act_n] - util[ob, act]
            #util = td_zero(util, ob, act, delta)
            util, eligible_mat = td_lambda(util, ob, act, delta, eligible_mat, eligible_factor)
            ob = ob_n
        rewards = [run(env, util) for j in range(10)]
        if all(map(lambda x: x>0, rewards)):
            # convergence detected if rewards are all positive
            print("finish training earlier using {0} iteration".format(i))
            print("final rewards: {0}".format(rewards))
            break

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
    return total_r

if __name__ == "__main__":
    env = gym.make('Taxi-v2')
    #env = gym.make('Copy-v0')
    print(env.action_space)
    #env = gym.make('SpaceInvaders-v0')
    start_time = time.time()
    util = train(env)
    print("finish training using {0:.2f} sec".format(time.time()-start_time))
    for _ in range(3):
        run(env, util, render=True)
