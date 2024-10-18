from ch9_ActorCritic import ActorCritic
import gym
import torch
from torch.distributions import Categorical
if __name__=="__main__":
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    # model.load_state_dict(torch.load("/home/ssu20/rlDir/policy/ActorCritic/ActriCritic_BestModel_epi_00020_21.35.pth"))
    model.load_state_dict(torch.load("/home/ssu20/rlDir/policy/ActorCritic/ActriCritic_BestModel_epi_01780_1497.35.pth"))
    done = False
    s, _ = env.reset()
    score = 0
    while not done:
        prob = model.pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample().item()
        s_prime, r, done, truncated, info = env.step(a)
        score +=r
        s = s_prime
        print(score,a)
