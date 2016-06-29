#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import random
import argparse
import time
import json
import numpy as np
import cv2
from collections import deque
import pygame

from ale_util import logger
from ale_net import DLNetwork
from ale_interface import AleInterface

pygame.init()


class DQNLearning(object):
    def __init__(self, game_name, args):

        self.game_name = game_name
        self.logger = logger

        self.game = AleInterface(game_name, args)
        self.actions = self.game.get_actions_num()

        # DQN parameters
        self.observe = args.observe
        self.explore = args.explore
        self.replay_memory = args.replay_memory
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.init_epsilon = args.init_epsilon
        self.final_epsilon = args.final_epsilon
        self.save_model_freq = args.save_model_freq

        self.update_frequency = args.update_frequency
        self.action_repeat = args.action_repeat

        self.frame_seq_num = args.frame_seq_num
        if args.saved_model_dir == "":
            self.param_file = "./saved_networks/%s.json" % game_name
        else:
            self.param_file = "%s/%s.json" % (args.saved_model_dir, game_name)

        self.net = DLNetwork(game_name, self.actions, args)

        # screen parameters
        # self.screen = (args.screen_width, args.screen_height)
        # pygame.display.set_caption(game_name)
        # self.display = pygame.display.set_mode(self.screen)

        self.deque = deque()

    def param_serierlize(self, epsilon, step):
        json.dump({"epsilon": epsilon, "step": step}, open(self.param_file, "w"))

    def param_unserierlize(self):
        if os.path.exists(self.param_file):
            jd = json.load(open(self.param_file, 'r'))
            return jd['epsilon'], jd["step"]
        else:
            return self.init_epsilon, 0

    def process_snapshot(self, snap_shot):
        # rgb to gray, and resize
        snap_shot = cv2.cvtColor(cv2.resize(snap_shot, (80, 80)), cv2.COLOR_BGR2GRAY)
        # image binary
        # _, snap_shot = cv2.threshold(snap_shot, 1, 255, cv2.THRESH_BINARY)
        return snap_shot

    def show_screen(self, np_array):
        return
        # np_array = cv2.resize(np_array, self.screen)
        # surface = pygame.surfarray.make_surface(np_array)
        # surface = pygame.transform.rotate(surface, 270)
        # rect = pygame.draw.rect(self.display, (255, 255, 255), (0, 0, self.screen[0], self.screen[1]))
        # self.display.blit(surface, rect)
        # pygame.display.update()

    def train_net(self):
        # training
        max_reward = 0
        epsilon, global_step = self.param_unserierlize()
        step = 0
        epoch = 0
        while True:  # loop epochs
            epoch += 1
            # initial state
            self.game.reset_game()
            # initial state sequences
            state_seq = np.empty((80, 80, self.frame_seq_num))
            for i in range(self.frame_seq_num):
                state = self.game.get_screen_rgb()
                self.show_screen(state)
                state = self.process_snapshot(state)
                state_seq[:, :, i] = state
            stage_reward = 0
            while True:  # loop game frames
                # select action
                best_act = self.net.predict([state_seq])[0]
                if random.random() <= epsilon or len(np.unique(best_act)) == 1:  # random select
                    action = random.randint(0, self.actions - 1)
                else:
                    action = np.argmax(best_act)
                # carry out selected action
                reward_n = self.game.act(action)
                state_n = self.game.get_screen_rgb()
                self.show_screen(state)
                state_n = self.process_snapshot(state_n)
                state_n = np.reshape(state_n, (80, 80, 1))
                state_seq_n = np.append(state_n, state_seq[:, :, : (self.frame_seq_num - 1)], axis=2)
                terminal_n = self.game.game_over()
                # scale down epsilon
                if step > self.observe and epsilon > self.final_epsilon:
                    epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore
                # store experience
                act_onehot = np.zeros(self.actions)
                act_onehot[action] = 1
                self.deque.append((state_seq, act_onehot, reward_n, state_seq_n, terminal_n))
                if len(self.deque) > self.replay_memory:
                    self.deque.popleft()
                # minibatch train
                if step > self.observe and step % self.update_frequency == 0:
                    for _ in xrange(self.action_repeat):
                        mini_batch = random.sample(self.deque, self.batch_size)
                        batch_state_seq = [item[0] for item in mini_batch]
                        batch_action = [item[1] for item in mini_batch]
                        batch_reward = [item[2] for item in mini_batch]
                        batch_state_seq_n = [item[3] for item in mini_batch]
                        batch_terminal = [item[4] for item in mini_batch]
                        # predict
                        target_rewards = []
                        batch_pred_act_n = self.net.predict(batch_state_seq_n)
                        for i in xrange(len(mini_batch)):
                            if batch_terminal[i]:
                                t_r = batch_reward[i]
                            else:
                                t_r = batch_reward[i] + self.gamma * np.max(batch_pred_act_n[i])
                            target_rewards.append(t_r)
                        # train Q network
                        self.net.fit(batch_state_seq, batch_action, target_rewards)
                # update state
                state_seq = state_seq_n
                step += 1
                # serierlize param
                # save network model
                if step % self.save_model_freq == 0:
                    global_step += step
                    self.param_serierlize(epsilon, global_step)
                    self.net.save_model("%s-dqn" % self.game_name, global_step=global_step)
                    self.logger.info("save network model, global_step=%d, cur_step=%d" % (global_step, step))
                # state description
                if step < self.observe:
                    state_desc = "observe"
                elif epsilon > self.final_epsilon:
                    state_desc = "explore"
                else:
                    state_desc = "train"
                # record reward
                print "game running, step=%d, action=%s, reward=%d, max_Q=%.6f, min_Q=%.6f" % \
                          (step, action, reward_n, np.max(best_act), np.min(best_act))
                if reward_n > stage_reward:
                    stage_reward = reward_n
                if terminal_n:
                    break
            # record reward
            if stage_reward > max_reward:
                max_reward = stage_reward
            self.logger.info(
                "epoch=%d, state=%s, step=%d(%d), max_reward=%d, epsilon=%.5f, reward=%d, max_Q=%.6f" %
                (epoch, state_desc, step, global_step, max_reward, epsilon, stage_reward, np.max(best_act)))

    def play_game(self, epsilon):
        # play games
        max_reward = 0
        epoch = 0
        if epsilon == 0.0:
            epsilon, _ = self.param_unserierlize()
        while True:  # epoch
            epoch += 1
            self.logger.info("game start...")
            # init state
            self.game.reset_game()
            state_seq = np.empty((80, 80, self.frame_seq_num))
            for i in range(self.frame_seq_num):
                state = self.game.get_screen_rgb()
                self.show_screen(state)
                state = self.process_snapshot(state)
                state_seq[:, :, i] = state
            stage_step = 0
            stage_reward = 0
            while True:
                # select action
                best_act = self.net.predict([state_seq])[0]
                if random.random() < epsilon or len(np.unique(best_act)) == 1:
                    action = random.randint(0, self.actions - 1)
                else:
                    action = np.argmax(best_act)
                # carry out selected action
                reward_n = self.game.act(action)
                state_n = self.game.get_screen_rgb()
                self.show_screen(state_n)
                state_n = self.process_snapshot(state_n)
                state_n = np.reshape(state_n, (80, 80, 1))
                state_seq_n = np.append(state_n, state_seq[:, :, : (self.frame_seq_num - 1)], axis=2)
                terminal_n = self.game.game_over()

                state_seq = state_seq_n
                # record
                if reward_n > stage_reward:
                    stage_reward = reward_n
                if terminal_n:
                    time.sleep(2)
                    break
                else:
                    stage_step += 1
                    stage_reward = reward_n
                    print "game running, step=%d, action=%d, reward=%d" % \
                          (stage_step, action, reward_n)
            # record reward
            if stage_reward > max_reward:
                max_reward = stage_reward
            self.logger.info("game over, epoch=%d, step=%d, reward=%d, max_reward=%d" %
                             (epoch, stage_step, stage_reward, max_reward))


def parser_argument():
    parse = argparse.ArgumentParser()
    parse.add_argument("--play", action="store", help="play games, you can specify model file in model directory")
    parse.add_argument("--train", action="store", help="train DQNetwork, game names is needed")
    parse.add_argument("--gpu", action="store", default=0, help="specify gpu number")
    args = parse.parse_args()
    gpu = int(args.gpu)
    if args.play is not None:
        if not args.play.isdigit():
            game_name = args.play
        else:
            game_name = "breakout"
        dqn = DQNLearning(game_name, gpu=None)
        dqn.play_game()
    elif args.train is not None:
        if not args.train.isdigit():
            game_name = args.train
        else:
            game_name = "breakout"
        dqn = DQNLearning(game_name, gpu=gpu)
        dqn.train_net()
    else:
        parse.print_help()


if __name__ == "__main__":
    parser_argument()
