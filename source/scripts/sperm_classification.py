from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.applications.densenet import preprocess_input


class SpermClassification(object):
	def __init__(self, model_path: str) -> None:
		"""
		"""
		self.model = tf.keras.models.load_model(model_path)
		print(self.model.summary())

	def __call__(self, input:list, batch_size:int=96) -> Any:
		list_batch_images = self.init_batch(input)
		list_predict = []
		for batch in list_batch_images:
			y_predict = self.model.predict(batch)
			y_target = np.argmax(y_predict, axis=1)
			list_predict.extend(y_target)

		return list_predict
	
	def preprocess(self, input) -> Any:
		"""
		preprocess image
		"""
		img = cv2.resize(input, dsize=(150, 150))
  
		return preprocess_input(image.img_to_array(img))
	
	def init_batch(self, inputs, batch_size:int=96) -> Any:
		list_batchs = []
		batch = []
		
		for i, image in enumerate(inputs, start=0):
			img_convert = self.preprocess(image)
			if i % batch_size != 0:
				batch.append(img_convert)
			else:
				batch.append(img_convert)
				list_batchs.append(np.array(batch))
				batch = []
		
		if batch != []:
			list_batchs.append(np.array(batch))

		return list_batchs
