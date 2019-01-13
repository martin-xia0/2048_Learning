import os
import numpy as np


board_dic = {0:0, 2:1, 4:2, 8:3, 16:4, 32:5, 64:6, 128:7, 256:8, 512:9, 1024:10, 2048:11, 4096:12}
move_dic = {0:[1,0,0,0], 1:[0,1,0,0], 2:[0,0,1,0], 3:[0,0,0,1]}

new_board = []
for unit in range(180):
	board_dir = './new_data/board_{}.npy'.format(unit)
	for board in np.load(board_dir):
		new_board.append(np.array([board_dic[i] for i in board.flatten()]).reshape(4,4))
	print('unit {} size {}'.format(unit, len(new_board)))
#for unit in range(150,161):
#	board_dir = './new_data/board_{}.npy'.format(unit)
#	for board in np.load(board_dir):
#		new_board.append(np.array([board_dic[i] for i in board.flatten()]).reshape(4,4))
#	print('unit {} size {}'.format(unit, len(new_board)))
#for unit in range(200,226):
#	board_dir = './new_data/board_{}.npy'.format(unit)
#	for board in np.load(board_dir):
#		new_board.append(np.array([board_dic[i] for i in board.flatten()]).reshape(4,4))
#	print('unit {} size {}'.format(unit, len(new_board)))
new_board = np.array(new_board)
print('finish, shape {}, last {}'.format(new_board.shape, new_board[-1]))
np.save('./rnn_data/board_{}_5'.format('train'), new_board)

new_move = []
for unit in range(180):
	move_dir = './new_data/move_{}.npy'.format(unit)
	for move in np.load(move_dir):
		new_move.append(np.array([move_dic[i] for i in move.flatten()]).reshape(4))
	print('unit {} size {}'.format(unit, len(new_move)))
#for unit in range(150,161):
#	move_dir = './new_data/move_{}.npy'.format(unit)
#	for move in np.load(move_dir):
#		new_move.append(np.array([move_dic[i] for i in move.flatten()]).reshape(4))
#	print('unit {} size {}'.format(unit, len(new_move)))
#for unit in range(200,226):
#	move_dir = './new_data/move_{}.npy'.format(unit)
#	for move in np.load(move_dir):
#		new_move.append(np.array([move_dic[i] for i in move.flatten()]).reshape(4))
#	print('unit {} size {}'.format(unit, len(new_move)))
new_move = np.array(new_move)
print('finish, shape {}, last {}'.format(new_move.shape, new_move[-1]))
np.save('./rnn_data/move_{}_5'.format('train'), new_move)