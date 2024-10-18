from ch9_REINFORCE import Policy
import gym
import torch
from torch.distributions import Categorical
if __name__=="__main__":
    env = gym.make('CartPole-v1',render_mode='human')
    pi = Policy()
    pi.load_state_dict(torch.load("./policy/Reinforce/Reinforce_BestModel_epi_01100_200.75.pth"))
    s, _ = env.reset()
    # env.render(mode='human')
    done = False
    while not done: # CartPole-v1 forced to terminates at 500 step.
        prob = pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample()
        s_prime, r, done, truncated, info = env.step(a.item())
        env.render()
        s = s_prime
        print(r)