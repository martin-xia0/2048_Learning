from game2048.game import Game
from game2048.displays import Display
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import os
import logging

def single_run(board_data, move_data, size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(board_data=board_data, move_data=move_data, game=game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 1024
    N_TESTS = 10
    logging.basicConfig(filename='get_data.log', format='%(asctime)s%(levelname)s%(message)s')
    logger = logging.getLogger('get_data')
    hdlr = logging.FileHandler('get_data.log')
    formatter = logging.Formatter('%(asctime)s%(levelname)s%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    '''====================
    Use your own agent here.'''
    from game2048.agents import DataAgent as TestAgent
    '''===================='''
    turn = 0
    unit_num = 0
    board_data = []
    move_data = []
    while True:
        turn += 1
        scores = []
        score = single_run(board_data, move_data, GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)
        data_size = len(board_data)
        if data_size > 100:
            board = np.array(board_data)
            move = np.array(move_data)
            info = 'unit {}, board shape {} move shape {}'.format(unit_num, board.shape, move.shape)
            np.save('./new_data/board_{}.npy'.format(unit_num), np.array(board_data))
            np.save('./new_data/move_{}.npy'.format(unit_num), np.array(move_data))
            logger.info(info)
            board_data = []
            move_data = []
            unit_num += 1
        info = "turn: {}".format(turn)
        logger.info(info)
        
