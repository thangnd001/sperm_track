from typing import Any

import numpy as np
import tensorflow as tf


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

		return input
	
	def init_batch(self, inputs, batch_size:int=96) -> Any:
		list_batchs = []
		batch = []
		
		for i, image in enumerate(inputs, start=0):
			if i % batch_size != 0:
				batch.append(image)
			else:
				batch.append(image)
				list_batchs.append(batch)
				batch = []
		
		if batch != []:
			list_batchs.append(batch)

		return list_batchs
