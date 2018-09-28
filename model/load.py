import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf


def init(): 
	with open('model.json','r') as json_file:
		loaded_model_json = json_file.read()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("model_weights.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='mse',optimizer='rmsprop')
	graph = tf.get_default_graph()

	return loaded_model,graph