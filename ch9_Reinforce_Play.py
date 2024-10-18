from ch9_REINFORCE import Policy
import gym
import torch
from torch.distributions import Categorical
if __name__=="__main__":
    env = gym.make('CartPole-v1')
    pi = Policy()
    pi.load_state_dict(torch.load("./policy/Reinforce/Reinforce_BestModel_epi_03820_1333.65.pth"))
    s, _ = env.reset()
    done = False
    while not done: # CartPole-v1 forced to terminates at 500 step.
        prob = pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample()
        s_prime, r, done, truncated, info = env.step(a.item())
        s = s_prime