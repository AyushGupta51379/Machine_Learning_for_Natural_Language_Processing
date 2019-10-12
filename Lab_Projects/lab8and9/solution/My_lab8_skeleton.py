from __future__ import print_function
import collections
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import argparse
import math
from data_helper import load_data
import os

# os.environ['CUDA_VISIBLE_DEVICES']="" # uncomment this line, if you use cpu

class TestCallback(Callback):
	"""
	Calculate Perplexity
	"""
	def __init__(self, test_data, model):
		self.test_data = test_data
		self.model = model
	def on_epoch_end(self, epoch, logs={}):
		x, y = self.test_data
		x_probs = self.model.predict(x)
		ppl = self.evaluate_batch_ppl(x_probs,y)
		print('\nValidation Set Perplexity: {0:.2f} \n'.format(ppl))
	def evaluate_ppl(self, x, y):
		x = x.reshape(-1, x.shape[-1])
		y = y.reshape(-1)
		return np.exp(np.mean(-np.log(np.diag(x[:, y]))))
	def evaluate_batch_ppl(self, x, y):
		eval_batch_size = 8
		x = x.reshape(-1, x.shape[-1])
		y = y.reshape(-1)
		ppl = 0.0
		for i in range(math.ceil(len(x)/eval_batch_size)):
			batch_x = x[i*eval_batch_size:(i+1)*eval_batch_size,:]
			batch_y = y[i*eval_batch_size:(i+1)*eval_batch_size]
			ppl += np.sum(np.log(np.diag(batch_x[:, batch_y])))
		return np.exp(-ppl/x.shape[0])

print('Loading data')
x_train, y_train, x_valid, y_valid, vocabulary_size = load_data()

print(x_train.shape)
print(y_train.shape)

num_training_data = x_train.shape[0]
sequence_length = x_train.shape[1]

print('Vocab Size',vocabulary_size)


# training parameters
drop = 0.5
epochs = 1
batch_size = 8
embedding_dim = 10
print('epoch ', epochs)
# lstm parameters
hidden_size = 10

inputs = Input(shape=(sequence_length,), dtype='int32')
# inputs -> [batch_size, sequence_length]

emb_layer = Embedding(input_dim=vocabulary_size, 
					output_dim=embedding_dim, 
					input_length=sequence_length)
# emb_layer.trainable = False
# if you uncomment this line, the embeddings will be untrainable

embedding = emb_layer(inputs)
# embedding -> [batch_size, sequence_length, embedding_dim]

drop_embed = Dropout(drop)(embedding) 
# dropout at embedding layer

# add a LSTM here, set units=hidden_size, return_sequences=True
# please read https://keras.io/layers/recurrent/
# output: lstm_out_1 -> [batch_size, sequence_length, hidden_size]
lstm_out_1 = LSTM(units=hidden_size,return_sequences=True)(drop_embed)

# add a TimeDistributed here, set layer = Dense(units=vocabulary_size,activation='softmax')
# please read  https://keras.io/layers/wrappers/
# output: outputs -> [batch_size, sequence_length, vocabulary_size]
outputs = TimeDistributed(Dense(units=vocabulary_size,
	activation='softmax'))(lstm_out_1)

# End of Model Architecture
# ----------------------------------------#


model = Model(inputs=inputs, outputs=outputs)

adam = Adam()
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

print(model.summary())

print("Traning Model...")
history = model.fit(x_train, y_train, 
		batch_size=batch_size, 
		epochs=epochs,
		verbose=1,
		callbacks=[TestCallback((x_valid,y_valid),model=model)])

'''
if you use windows operating system, set verbose=0 in history = model.fit(...), uncomment the following lines.
Then there is no progress bar, it will return you the losses after the training finish.
'''
# for i in range(epochs):
# 	print('Epoch {0}/{1} Loss: {2:.4f}\n'.format(i+1, epochs, history.history['loss'][i] )) 


