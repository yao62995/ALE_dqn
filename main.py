#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import argparse
from ale_learning import DQNLearning

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def parser_argument():
    parse = argparse.ArgumentParser()
    parse.add_argument("--game", type=str, help="game name")
    parse.add_argument("--handle", type=str, help="\"train\" or \"play\"")

    # env_args = parse.add_argument_group("ALE_Interface")
    parse.add_argument("--display_screen", type=str2bool, default=True, help="whether to display screen")
    parse.add_argument("--frame_skip", type=int, default=4, help="frame skip number")
    parse.add_argument("--repeat_action_probability", type=float, default=0, help="repeat action probability")
    parse.add_argument("--color_averaging", type=str2bool, default=True, help="color average")
    parse.add_argument("--random_seed", type=int, default=0, help="random seed")

    # learn_args = parse.add_argument_group("ALE_Learning")
    parse.add_argument("--observe", type=int, default=10000, help="number of random start")
    parse.add_argument("--explore", type=float, default=2000000.0, help="")
    parse.add_argument("--replay_memory", type=int, default=50000, help="")
    parse.add_argument("--gamma", type=float, default=0.99, help="")
    parse.add_argument("--init_epsilon", type=float, default=1.0, help="")
    parse.add_argument("--final_epsilon", type=float, default=0.1, help="")
    parse.add_argument("--update_frequency", type=int, default=4, help="")
    parse.add_argument("--action_repeat", type=int, default=4, help="")

    # net_args = parse.add_argument_group("ALE_Network")
    parse.add_argument("--device", type=str, default="gpu", help="cpu or gpu")
    parse.add_argument("--gpu", type=int, default=0, help="gpu average")
    parse.add_argument("--batch_size", type=int, default=32, help="batch size")
    parse.add_argument("--optimizer", choices=['rmsprop', 'adam', 'sgd'], default='rmsprop', help='Network optimization algorithm')
    parse.add_argument("--learn_rate", type=float, default=0.00025, help="Learning rate")
    parse.add_argument("--decay_rate", type=float, default=0.95, help="decay rate, used for Rmsprop")
    parse.add_argument("--momentum", type=float, default=0.95, help="momentum, used for Rmsprop")

    parse.add_argument("--with_pool_layer", type=str2bool, default=False, help="whether has max_pool layer")
    parse.add_argument("--frame_seq_num", type=int, default=4, help="frame seq number")
    parse.add_argument("--saved_model_dir", type=str, default="", help="")
    parse.add_argument("--model_file", type=str, default="", help="")
    parse.add_argument("--save_model_freq", type=int, default=100000, help="")

    # parse.add_argument("--screen_width", type=int, default=80, help="resize screen width")
    # parse.add_argument("--screen_height", type=int, default=80, help="resize screen height")

    # parameter for play
    parse.add_argument("--play_epsilon", type=float, default=0.0, help="a float value in [0, 1), 0 means use global train epsilon")

    args = parse.parse_args()
    if args.game is None or args.handle is None:
        parse.print_help()
    if args.handle != "train":  # use cpu when play games
        args.device = "cpu"
    dqn = DQNLearning(args.game, args)
    if args.handle == "train":
        dqn.train_net()
    else:
        dqn.play_game(args.play_epsilon)


if __name__ == "__main__":
    parser_argument()