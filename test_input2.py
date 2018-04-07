# -*- coding: utf-8 -*-

from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import numpy as np
#import random
import pygame
from skimage import transform,exposure
from collections import deque

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam




OBSERVATION = 3200 # timesteps to observe before training
INITIAL_EPSILON = 0.1 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon

class Bot():
    """
            This is our Test agent. It's gonna pick some actions after training!
    """
    def __init__(self, lr):
        
        self.lr = lr
        self.game = Pixelcopter(width=480, height=480)
        self.p = PLE(self.game, fps=60, display_screen=True)
        self.actions = self.p.getActionSet()

        
    #def pickAction(self, reward, obs):
     #   return random.choice(self.actions)

    def frame_step(act_inp):
        terminal=False
        reward = self.p.act(act_inp)
        if self.p.game_over():
            self.p.reset_game()
            terminal=True
            reward = -1
        else:
            reward = 1
        
        self.score = self.p.getScore()
        img = self.p.getScreenGrayscale()
        img = transform.resize(img,(80,80))
        img = np.ravel(exposure.rescale_intensity(img, out_range=(0, 255)))
        
        return img,reward, terminal
    
    def build_model(self):
        print("Building the model..")
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))
   
        adam = Adam(lr=self.lr)
        model.compile(loss='mse',optimizer=adam)
        self.model = model
        print("Finished building the model..")
    
    
    def trainNetwork(self,mode):
        D = deque()
        
        x_t, r_0, terminal = self.frame_step(self.actions[0])
        x_t = x_t / 255.0

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        #print (s_t.shape)

        #need to reshape for keras
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4
        
        if mode == 'Run':
            OBSERVE = 999999999    #We keep observe, never train
            epsilon = FINAL_EPSILON
            print ("Now we load weight")
            self.model.load_weights("model.h5")
            adam = Adam(lr=self.lr)
            self.model.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")    
        else:                       #We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        
        
        
        
        
img_rows , img_cols = 80, 80
img_channels = 4 #We stack 4 frames
lr = 0.01




reward = 0.0

i_size = 0
in_arr = []
temp_in = []
for i in range(100): #no of frames
   if p.game_over():
           p.reset_game()
   if i_size == 3:
       #print (temp_in.shape)
       in_arr.append(temp_in)
       temp_in = []
       
   else:
        img = p.getScreenGrayscale()
        img = transform.resize(img,(80,80))
        img = np.ravel(exposure.rescale_intensity(img, out_range=(0, 255)))
        #print (img.shape)
        temp_in.append(img)
        
   observation = p.getScreenRGB()
   action = agent.pickAction(reward, observation)
   reward = p.act(action)
   
#img = p.getScreenGrayscale()
pygame.quit()
# -*- coding: utf-8 -*-


#print (in_arr[0][1][0])
