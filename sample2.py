import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    # 定数を定義
    LEFT = 0
    RIGHT = 1
    env = gym.make('CartPole-v0')
    for i in range(10):
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = 0
            if observation[1]<-0.5:
                action = 1
            elif observation[1]>0.5:
                action = 0
            elif observation[2]>0:
                action = 1
            #observation, reward, done, info = env.step(env.action_space.sample())
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode{} finished after {} timesteps".format(i, t+1))
                break
    env.close()

if __name__ == '__main__':
    main()