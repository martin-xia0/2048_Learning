import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import os
from game2048.game import Game


def xzx_board_to_move(models, board, score):
	board_dic = {0:0, 2:1, 4:2, 8:3, 16:4, 32:5, 64:6, 128:7, 256:8, 512:9, 1024:10, 2048:11, 4096:12}
	
	board = np.array([board_dic[i] for i in board.flatten()]).reshape(4,4)



	move = []
	intu = intuition(board)
	move.append(intu)
	for i in range(4):
		board_rot = np.rot90(board,i)
		# # model 13
		# board_rot = board_rot.reshape(1,4,4,1)
		# # output move dirction
		# pre_move = models[0].predict(board_rot).reshape(4)
		# move1 = (np.argmax(pre_move,axis=0)+4-i) % 4
		# move.append(move1)
		# model 4
		board_rot = board_rot.reshape(1,4,4)
		# output move dirction
		pre_move = models[0].predict(board_rot).reshape(4)
		move2 = (np.argmax(pre_move,axis=0)+4-i) % 4
		# print('move1: {} move2: {}'.format(move1, move2))
		move.append(move2)
	# print(move)


	result = [move.count(i) for i in range(4)]
	direction = result.index(max(result))
	return direction


def intuition(board):
    # ------------------------------
    # 正常情况，使用直觉快速思考（近似贪心法）
    score_dict = {0:0,1:0,2:0,3:0}
    game = Game(4, 4096, enable_rewrite_board=True)
    for i in range(4):
        # 每次模拟前置零
        game.board = board
        game.score_move = 0
        game.only_move(i)
        # 无合并，就是菜
        # print(game.board)
        if game.score_move == 0:
            score_dict[i] = -100 + board_score(game.board)
        # 有合并，计算权值
        else:
            score_dict[i] += game.score_move + board_score(game.board)
    # print('score_dict {}'.format(score_dict))
    direction = max(score_dict, key=score_dict.get)
    return direction


# 计算棋盘的评估分
def board_score(board):

    # 1.空格数
    space_score = list(board.flatten()).count(0)
    # print('space_score {}'.format(space_score))

    # 2.单调性(平滑度)
    monotone_score = 0
    board_rot = np.rot90(board, 1)
    # 共获得4行4列8个评分
    for row in board:
        row = [i for i in row if i != 0]
        if len(row) < 2:
            monotone_row = 0
        elif len(row) == 2:
            monotone_row = abs(row[1]-row[0])
        elif len(row) == 3:
            monotone_row = abs(row[2]-row[1])+abs(row[1]-row[0])
        elif len(row) == 4:
            monotone_row = abs(row[3]-row[2])+abs(row[2]-row[1])+abs(row[1]-row[0])
        monotone_score += monotone_row
        # print('monotone_row {}'.format(monotone_row))
    for col in board_rot:
        col = [i for i in col if i != 0]
        if len(col) < 2:
            monotone_col = 0
        elif len(col) == 2:
            monotone_col = abs(col[1]-col[0])
        elif len(col) == 3:
            monotone_col = abs(col[2]-col[1])+abs(col[1]-col[0])
        elif len(col) == 4:
            monotone_col = abs(col[3]-col[2])+abs(col[2]-col[1])+abs(col[1]-col[0])
        monotone_score += monotone_col
        # print('monotone_col {}'.format(monotone_col))
    monotone_score = monotone_score/8
    # print('monotone_score {}'.format(monotone_score))

    # 3.最大值
    max_score = max(board.flatten())
    # print('max_score {}'.format(max_score))

    # 4.良拐角 如果角点是最大值加分
    corner_score = 0
    corner = [board[0][0], board[0][3], board[3][0], board[3][3]]
    corner_score = corner.count(max(board.flatten()))
    # print('corner_score {}'.format(corner_score))

    # 按比例掺杂,获得总成绩
    score = 0
    score = space_score*10 - monotone_score*2 + max_score + corner_score*6
    # print('board score {}'.format(score))
    return score