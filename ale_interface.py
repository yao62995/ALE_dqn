#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import sys
from ale_python_interface import ALEInterface

import pygame
pygame.init()

class AleInterface(object):
    def __init__(self, game, args):
        self.game = game
        self.ale = ALEInterface()

        # if sys.platform == 'darwin':
        #     self.ale.setBool('sound', False)  # Sound doesn't work on OSX
        # elif sys.platform.startswith('linux'):
        #     self.ale.setBool('sound', True)
        # self.ale.setBool('display_screen', True)
        #
        self.ale.setBool('display_screen', args.display_screen)

        self.ale.setInt('frame_skip', args.frame_skip)
        self.ale.setFloat('repeat_action_probability', args.repeat_action_probability)
        self.ale.setBool('color_averaging', args.color_averaging)
        self.ale.setInt('random_seed', args.random_seed)

        #
        # if rand_seed is not None:
        #     self.ale.setInt('random_seed', rand_seed)

        rom_file = "./roms/%s.bin" % game
        if not os.path.exists(rom_file):
            print "not found rom file:", rom_file
            sys.exit(-1)
        self.ale.loadROM(rom_file)

        self.actions = self.ale.getMinimalActionSet()


    def get_actions_num(self):
        return len(self.actions)

    def act(self, action):
        reward = self.ale.act(self.actions[action])
        return reward

    def get_screen_gray(self):
        return self.ale.getScreenGrayscale()

    def get_screen_rgb(self):
        return self.ale.getScreenRGB()

    def game_over(self):
        return self.ale.game_over()

    def reset_game(self):
        return self.ale.reset_game()



