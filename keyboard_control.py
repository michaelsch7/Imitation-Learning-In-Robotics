import sys
import numpy as np
import pandas as pd
import pygame
import metaworld
from pygame.locals import KEYDOWN, QUIT
from PIL import Image
import matplotlib.pyplot as plt

actionsList = []
rewardList = []
gripOpeness = []
gripSpace = []
goodtogo = True

pygame.init()
screen = pygame.display.set_mode((400, 300))


char_to_action = {
    "w": np.array([0, -1, 0, 0]),
    "a": np.array([1, 0, 0, 0]),
    "s": np.array([0, 1, 0, 0]),
    "d": np.array([-1, 0, 0, 0]),
    "q": np.array([1, -1, 0, 0]),
    "e": np.array([-1, -1, 0, 0]),
    "z": np.array([1, 1, 0, 0]),
    "c": np.array([-1, 1, 0, 0]),
    "f": np.array([0, 0, 1, 0]),
    "g": np.array([0, 0, -1, 0]),
    "r": "close",
    "t": "open",
    "x": "toggle",
    "m": "reset",
    "p": "put obj in hand",
}

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
timestamp = 0

while True and goodtogo:
    
    # Save image at timestamp as file
    env.render_mode = 'rgb_array'
    frame = env.render()
    env.render_mode = 'human'
    img = Image.fromarray(frame)
    
    img.save(f'Run3/enviroment{timestamp}.png')
    timestamp = timestamp+1
    
    # Run action 
    if not lock_action:
       action[:3] = 0
    if not random_action:
        for event in pygame.event.get():
            event_happened = True
            if event.type == QUIT:
                sys.exit()
            if event.type == KEYDOWN:
                char = event.dict["key"]
                new_action = char_to_action.get(chr(char), None)
                if str(new_action) == "toggle":
                    lock_action = not lock_action
                elif str(new_action) == "reset":
                    done = True
                elif str(new_action) == "close":
                    action[3] = 1
                elif str(new_action) == "open":
                    action[3] = -1
                elif new_action is not None and isinstance(new_action, np.ndarray):
                    action[:3] = new_action[:3]
                else:
                    action = np.zeros(3)
    else:
        action = env.action_space.sample()

    
    actionsList.append(action.copy())
    ob, reward, f, done, info = env.step(action)
    rewardList.append(reward)
    gripOpeness.append(ob[3])
    gripSpace.append(ob[0:3])
    
    if done:
        np.savetxt("Run3/Actions-move.csv", np.array(actionsList).astype(float).astype(int))
        np.savetxt("Run3/rewards.csv", rewardList)
        np.savetxt("Run3/gripperOpeness.csv", gripOpeness)
        np.savetxt("Run3/gripperPosition.csv", gripSpace)
        actionsList.clear()
        rewardList.clear()
        gripOpeness.clear()
        gripSpace.clear()
        goodtogo = False

        obs = env.reset()
        
    env.render()
