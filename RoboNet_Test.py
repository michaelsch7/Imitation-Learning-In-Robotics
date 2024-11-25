import sys
import numpy as np
import pandas as pd
import pygame
from pygame.locals import KEYDOWN, QUIT
import torch
import cv2
import metaworld
import random
from Model/RoboNet import RoboNet
from RoboNetUtilities import utils

actionsList = []
pygame.init()
screen = pygame.display.set_mode((400, 300))
end = False

char_to_action = {
    "w": np.array([0, -1, 0, 0]),
    "a": np.array([1, 0, 0, 0]),
    "s": np.array([0, 1, 0, 0]),
    "d": np.array([-1, 0, 0, 0]),
    "q": np.array([1, -1, 0, 0]),
    "e": np.array([-1, -1, 0, 0]),
    "z": np.array([1, 1, 0, 0]),
    "c": np.array([-1, 1, 0, 0]),
    "k": np.array([0, 0, 1, 0]),
    "j": np.array([0, 0, -1, 0]),
    "y": "close",
    "u": "open",
    "x": "toggle",
    "r": "reset",
    "p": "put obj in hand",
}
# Add new location for block 
ml1 = metaworld.ML1('pick-place-v2')
env = ml1.train_classes['pick-place-v2'](render_mode='human', camera_name='corner2')
env._partially_observable = False
env._freeze_rand_vec = False
env._set_task_called = True
env.reset()
env._freeze_rand_vec = True
lock_action = False
random_action = False
obs = env.reset()

action = np.zeros(4)
RoboNet = RoboNet(8)
RoboNet.load_state_dict(torch.load("ModelWeights/RoboNetWeights.pth")) 
arm_input = torch.tensor([0.1, 0.1, 0.1, 0.1]).unsqueeze(0) # Initilize arm input before first step (t=0)
RoboNet.eval()
timestep = 0
done = False
closed=False

while True: 
    env.render_mode = 'rgb_array'
    frame = env.render() # Save state in RGB array form as input for RoboNet
    env.render_mode = 'human'
    frame = cv2.flip(frame, -1)
    
    if(random.randint(1, 100) < 100):
        img_t = torch.from_numpy(frame).permute(2,0,1).float().unsqueeze(0)
        img_t = img_t / 255.0
        
        action = RoboNet(img_t, arm_input)
        action = np.argmax(action.detach().numpy())
        if not closed and action == 7: 
            closed=True

        
        action = utils().int2move(action, closed)
    else: 
        action = utils().int2move(random.randint(1,7), closed)
      
    ob, reward, f, done, info = env.step(action)
    
    arm_input = torch.tensor(np.array(ob[0:4])).float().unsqueeze(0)
    print(reward)
    env.render()
    timestep +=1
    if done:
        break
    
   
