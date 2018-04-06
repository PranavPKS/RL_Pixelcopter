# -*- coding: utf-8 -*-

from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import numpy as np
import pygame
from skimage import transform,exposure

class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """
    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]
        #return self.actions[1]
    
game = Pixelcopter(width=480, height=480)
p = PLE(game, fps=60, display_screen=True)
agent = NaiveAgent(p.getActionSet())



p.init()
reward = 0.0

i_size = 0
in_arr = []
temp_in = []
for i in range(100):
   if p.game_over():
           p.reset_game()
   if i_size == 3:
       #print (temp_in.shape)
       in_arr.append(temp_in)
       temp_in = []
       i_size = 0
   else:
        img = p.getScreenGrayscale()
        img = transform.resize(img,(80,80))
        img = np.ravel(exposure.rescale_intensity(img, out_range=(0, 255)))
        print (img.shape)
        temp_in.append(img)
        i_size += 1
   observation = p.getScreenRGB()
   action = agent.pickAction(reward, observation)
   reward = p.act(action)
   
#img = p.getScreenGrayscale()
pygame.quit()
# -*- coding: utf-8 -*-


print (in_arr[0][1][0])
