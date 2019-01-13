import numpy as np
import keras
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D
import os


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None, model=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False, webapp=None):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                score = self.game.score
                
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction

    def log_data(self, code_len, direction):
        board_file = "./raw_data_10/board_{}.txt".format(code_len)
        print('output board:{}'.format(board_file))
        move_file = "./raw_data_10/move_{}.txt".format(code_len)
        print('output move:{}'.format(move_file))
        
        # log board data
        board = self.game.board.reshape(16)
        onehot_board = ''
        for i in board:
            onehot_dot = bin(int(i))[2:].zfill(code_len)
            onehot_board += onehot_dot
        with open(board_file, 'a') as f:
            f.write(onehot_board+'\n')

        # log move direction data
        onehot_move = ['0001', '0010', '0100', '1000'][direction]
        with open(move_file, 'a') as f:
            f.write(onehot_move+'\n')

class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction

class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class MonteCarloAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .montecarlo import board_to_move_thread
        self.search_func = board_to_move_thread

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class XzxAgent(Agent):
    def __init__(self, models, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .xzx_model import xzx_board_to_move
        self.search_func = xzx_board_to_move
        self.models = models

    def step(self):
        direction = self.search_func(self.models, self.game.board,self.game.score)
        return direction

class OnlineAgent(Agent):
    def __init__(self, models, train_batch, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .xzx_model import xzx_board_to_move
        from .expectimax import board_to_move
        self.models = models
        self.expectmax_move = board_to_move
        self.train_batch = train_batch


    def step(self):
        direction = self.xzx_board_to_move()
        return direction


    def xzx_board_to_move(self):
        score = self.game.score
        board = self.game.board

        if score < 64:
            code_len = 6
        elif score < 128:
            code_len = 7
        elif score < 256:
            code_len = 8
        elif score < 512:
            code_len = 9 
        elif score < 1024:
            code_len = 10
        else:
            code_len = 12

        # choose model
        model = self.models[code_len]

        # input data
        board = board.reshape(16)
        print('board: {}'.format(board))
        board_input = []
        for dot in board:
            onehot_dot = bin(int(dot))[2:].zfill(code_len)
            for i in onehot_dot:
                board_input.append(int(i))
        board_input = np.array(board_input).reshape(1, 4, 4, code_len)
        print('input {}'.format(board_input.shape))

        # output move dirction
        move_out = model.predict(board_input).reshape(4)
        print('xzx move array: {}'.format(move_out))
        move = np.argmax(move_out, axis = 0)
        print('xzx move: {}'.format(move))

        # train model
        max_move =  self.expectmax_move(self.game.board)
        move_train = np.array([0, 0, 0, 0])
        move_train[max_move] = 1
        x = board_input
        y = move_train.reshape(1,4)
        history = model.train_on_batch(x, y)
        print('max move array {}'.format(move_train))
        print('max move {}'.format(max_move))
        print('history {}'.format(history))

        return move


class DataAgent(Agent):

    def __init__(self, board_data, move_data, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move
        self.board_data = board_data
        self.move_data = move_data

    def step(self):
        direction = self.search_func(self.game.board)
        self.board_data.append(self.game.board)
        self.move_data.append(direction) 
        return direction