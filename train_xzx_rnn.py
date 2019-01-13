import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential,load_model
from keras.layers import Conv2D,Concatenate,Flatten
from keras.layers import LSTM,BatchNormalization,Activation
from keras.layers import Dense,LeakyReLU
from keras.layers.core import Reshape
import logging

# # model 4
# # def build_model():
# 	model = Sequential()
# 	FILTERS = 256
# 	# 卷积层提取特征
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='valid',activation='relu',kernel_initializer='he_uniform'))
# 	model.add(BatchNormalization())
# 	model.add(Reshape((4,FILTERS),input_shape=(2,2,FILTERS)))
# 	# LSTM层处理CNN特征
# 	model.add(LSTM(units=300,return_sequences=True))
# 	# 两层全连接
# 	model.add(Dense(units=64,kernel_initializer='he_uniform'))
# 	model.add(Dense(units=4,kernel_initializer='he_uniform',activation='softmax'))

# 	# 建立模型
# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	return model



## model 7
# # def build_model():
# 	model = Sequential()
# 	FILTERS = 256
# 	# 卷积层提取特征
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='valid',activation='relu',kernel_initializer='he_uniform'))
# 	model.add(BatchNormalization())
# 	model.add(Reshape((4,FILTERS),input_shape=(2,2,FILTERS)))
# 	# LSTM层处理CNN特征
# 	model.add(LSTM(units=300,return_sequences=True))
# 	model.add(LSTM(units=64,return_sequences=True))
# 	model.add(LSTM(units=4,dropout=0.01,activation='softmax',return_sequences=False))
# 	# model.add(LSTM(units=64,activation='tanh',dropout=0.05))
# 	# 两层全连接
# 	# model.add(Dense(units=64,kernel_initializer='he_uniform'))
# 	# model.add(Dense(units=4,kernel_initializer='he_uniform',activation='softmax'))

# 	# 建立模型
# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	return model

# # model 8
# def build_model():
# 	model = Sequential()
# 	FILTERS = 128
# 	# 卷积层提取特征
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(1,3),padding='same',activation='relu',kernel_initializer='he_uniform'))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,1),padding='same',activation='relu',kernel_initializer='he_uniform'))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='valid',activation='relu',kernel_initializer='he_uniform'))
# 	model.add(BatchNormalization())
# 	model.add(Reshape((4,FILTERS),input_shape=(2,2,FILTERS)))
# 	# LSTM层处理CNN特征
# 	model.add(LSTM(units=300,return_sequences=True))
# 	model.add(LSTM(units=64,return_sequences=True))
# 	model.add(LSTM(units=4,dropout=0.01,activation='softmax',return_sequences=False))

# 	# 建立模型
# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	return model

# # model 9
# def build_model():
# 	model = Sequential()
# 	FILTERS = 128
# 	# 卷积层提取特征
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(1,3),padding='same',kernel_initializer='he_uniform'))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,1),padding='same',kernel_initializer='he_uniform'))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='valid',kernel_initializer='he_uniform'))
# 	model.add(Leaky_Relu(0.3))
# 	model.add(BatchNormalization())
# 	model.add(Reshape((4,FILTERS),input_shape=(2,2,FILTERS)))
# 	# LSTM层处理CNN特征
# 	model.add(LSTM(units=300,return_sequences=True))
# 	model.add(LSTM(units=64,return_sequences=True))
# 	model.add(LSTM(units=4,dropout=0.01,activation='softmax',return_sequences=False))
# 	# model.add(LSTM(units=64,activation='tanh',dropout=0.05))
# 	# 两层全连接
# 	# model.add(Dense(units=64,kernel_initializer='he_uniform'))
# 	# model.add(Dense(units=4,kernel_initializer='he_uniform',activation='softmax'))

# 	# 建立模型
# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	return model


# # model 10
# def build_model():
# 	model = Sequential()
# 	FILTERS = 128
# 	# 卷积层提取特征
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='valid',kernel_initializer='he_uniform'))
# 	model.add(LeakyReLU(0.3))
# 	model.add(BatchNormalization())
# 	model.add(Reshape((4,FILTERS),input_shape=(2,2,FILTERS)))

# 	# LSTM层处理CNN特征
# 	model.add(LSTM(units=256,return_sequences=True))
# 	model.add(LSTM(units=64,return_sequences=True))
# 	model.add(LSTM(units=16,dropout=0.01,return_sequences=True))
# 	model.add(LSTM(units=4,dropout=0.01,activation='softmax',return_sequences=False))

# 	# 建立模型
# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	return model


# # model 10
# def build_model():
# 	model = Sequential()
# 	FILTERS = 128
# 	# 卷积层提取特征
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='valid',kernel_initializer='he_uniform'))
# 	model.add(LeakyReLU(0.3))
# 	model.add(BatchNormalization())
# 	model.add(Reshape((4,FILTERS),input_shape=(2,2,FILTERS)))

# 	# LSTM层处理CNN特征
# 	model.add(LSTM(units=256,return_sequences=True))
# 	model.add(LSTM(units=64,return_sequences=True))
# 	model.add(LSTM(units=16,dropout=0.01,return_sequences=True))
# 	model.add(LSTM(units=4,dropout=0.01,activation='softmax',return_sequences=False))

# 	# 建立模型
# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	return model

# # model 11
# def build_model():
# 	model = Sequential()
# 	FILTERS = 512
# 	# 卷积层提取特征
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(1,3),padding='same',kernel_initializer='he_uniform'))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,1),padding='same',kernel_initializer='he_uniform'))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='same',kernel_initializer='he_uniform'))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(4,4),padding='valid',kernel_initializer='he_uniform'))
# 	model.add(LeakyReLU(0.3))
# 	model.add(BatchNormalization())
# 	model.add(Reshape((1,FILTERS),input_shape=(1,1,FILTERS)))

# 	# LSTM层处理CNN特征
# 	model.add(LSTM(units=128,return_sequences=True))
# 	model.add(LSTM(units=16,return_sequences=True))
# 	model.add(LSTM(units=4,dropout=0.01,activation='softmax',return_sequences=False))

# 	# 建立模型
# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	return model

# # model 13
# def build_model():
# 	model = Sequential()
# 	FILTERS = 512
# 	# 卷积层提取特征
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(1,4),padding='same',kernel_initializer='he_uniform'))
# 	model.add(LeakyReLU(0.3))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(4,1),padding='same',kernel_initializer='he_uniform'))
# 	model.add(LeakyReLU(0.3))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='same',kernel_initializer='he_uniform'))
# 	model.add(LeakyReLU(0.3))
# 	model.add(Conv2D(filters=FILTERS,kernel_size=(4,4),padding='valid',kernel_initializer='he_uniform'))
# 	model.add(LeakyReLU(0.3))
# 	model.add(BatchNormalization())
# 	model.add(Flatten())
# 	# model.add(Reshape((1,FILTERS),input_shape=(1,1,FILTERS)))
# 	# LSTM层处理CNN特征
# 	# model.add(LSTM(units=64,return_sequences=True))
# 	# model.add(LSTM(units=4,dropout=0.01,activation='softmax',return_sequences=False))
# 	model.add(Dense(units=64,kernel_initializer='he_uniform'))
# 	model.add(Dense(units=4,kernel_initializer='he_uniform',activation='softmax'))
# 	# 建立模型
# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	return model


# model 14
def build_model():
	model = Sequential()
	FILTERS = 256
	# 卷积层提取特征
	model.add(Conv2D(filters=FILTERS,kernel_size=(3,3),padding='valid',kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(Reshape((4,FILTERS),input_shape=(2,2,FILTERS)))
	# LSTM层处理CNN特征
	model.add(LSTM(units=300,dropout=0.01,return_sequences=True))
	model.add(LeakyReLU())
	model.add(Flatten())
	# 两层全连接
	model.add(Dense(units=64,kernel_initializer='he_uniform'))
	model.add(Dense(units=4,kernel_initializer='he_uniform',activation='softmax'))

	# 建立模型
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model




def load_data(label):
	board_dir = './rnn_data/board_{}_5.npy'.format(label)
	board = np.load(board_dir)
	# 输入一定要加一个时间参量，佛了
	board = board.reshape(board.shape[0],4,4,1)
	# print(board)
	move_dir = './rnn_data/move_{}_5.npy'.format(label)
	move = np.load(move_dir)
	# print("move:===",move)
	print('Finsh loading')
	print('Board {}'.format(board.shape))
	print('Move {}'.format(move.shape))
	return board, move


def train_model(model, epoch):
	board, move = load_data('train')
	model.fit(x=board,y=move,batch_size=64,verbose=1)
	return model

if __name__ == '__main__':
	logging.basicConfig(filename='train_rnn_14.log', format='%(asctime)s%(levelname)s%(message)s')
	logger = logging.getLogger('train_rnn_14')
	hdlr = logging.FileHandler('train_rnn_14.log')
	formatter = logging.Formatter('%(asctime)s%(levelname)s%(message)s')
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)
	logger.setLevel(logging.INFO)

	model = build_model()
	# model = load_model('./rnn_model/epoch_29.h5')
	K.set_value(model.optimizer.lr, 0.001)
	for epoch in range(30):
		# if (epoch > 10):
		print(K.get_value(model.optimizer.lr))
		if epoch > 10:
			K.set_value(model.optimizer.lr, 0.0001)
		# if epoch == 20:
		# 	K.set_value(model.optimizer.lr, 0.0001)
		model = train_model(model, epoch)
		model.save('./rnn_model/epoch_{}_14.h5'.format(epoch))

		test_board, test_move = load_data('test')
		# print("test_board=======",test_board[:10])
		print("test_move=======", test_move[:10])
		y_pred = model.predict(test_board[:10])
		print("pred:===",y_pred)
		print("pred:max===",np.argmax(y_pred,axis=1))
		# print(test_move)
		x = test_board[:-1]
		y = test_move[:-1]
		print(y)
		cost = model.evaluate(x=x,y=y,batch_size=64,verbose=1)
		print('epoch {}, result {}'.format(epoch, cost))
		logger.info('predict {} {}'.format(test_move[:10], np.argmax(y_pred,axis=1)))
		logger.info('epoch {}, result {}'.format(epoch, cost))