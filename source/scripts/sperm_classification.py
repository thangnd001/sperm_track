from typing import Any
import os

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.applications.densenet import preprocess_input

from sperm_classification.core.model_hsc_v1 import created_model_hsc_01
from sperm_classification.core.model import model_classification


class SpermClassification(object):
	def __init__(self, model_path: str, device: str, input_size:int = 40) -> None:
		"""
		"""
		# Set memory allocation
		gpus = tf.config.list_physical_devices('GPU')
		if gpus:
			# Create 2 virtual GPUs with 1GB memory each
			try:
				tf.config.set_logical_device_configuration(
					gpus[0],
					[tf.config.LogicalDeviceConfiguration(memory_limit=1024*5)])
				logical_gpus = tf.config.list_logical_devices('GPU')
				print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
			except RuntimeError as e:
				# Virtual devices must be set before GPUs have been initialized
				print(e)

		# Load model
		self.input_size = input_size
		self.device = device
		with tf.device(device):
			# Model current
			# self.model = tf.keras.models.load_model(model_path)

			# Model base
			self.model = model_classification(
				input_layer=(input_size,input_size,1),
				num_class=4,
				activation_dense='softmax',
				activation_block='LeakyReLU'
			)

			# Model hsm
			# self.model = created_model_hsc_01(
			# 	input_shape=(input_size,input_size,3),
			# 	number_class=4,
			# 	activation_dense='softmax',
			# 	activation_block='LeakyReLU'
			# )

			self.model.load_weights(model_path)

			print(self.model.summary())

	def __call__(self, input:list, batch_size:int=96) -> Any:
		list_batch_images = self.init_batch(input, self.input_size, batch_size)
		# print("INPUT = ", list_batch_images)
		list_predict = []
		with tf.device(self.device):
			for batch in list_batch_images:
				y_predict = self.model.predict(batch)
				# print("Y PREDICT = ", y_predict)
				y_target = np.argmax(y_predict, axis=1)
				
				# limit label
				y_target = np.clip(y_target, 0, 1)
				list_predict.extend(y_target)
		# print('LIST PREDICT = ', list_predict)
		print('TYPE _PREDICT = ', type(y_target))
		print("LIST PREDICT = ", list_predict)
		return list_predict
	
	def preprocess(self, input, input_size:int=150) -> Any:
		"""
		preprocess image
		"""
		img = input
		input_shape = img.shape[0]
		if input_size != input_shape:
			img = cv2.resize(input, dsize=(input_size, input_size))
		
		# Convert gray
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = np.expand_dims(img, axis=-1)
		
		if True and len(os.listdir("sperm_crop")) < 68:
			cv2.imwrite(f'sperm_crop/sperm_crop_{len(os.listdir("sperm_crop")) + 1}.png', img)
		# print("OK")
		# return preprocess_input(image.img_to_array(img))
		# return image.img_to_array(img, dtype=float) / 255.0
		return image.img_to_array(img, dtype=float)
	
	def init_batch(self, inputs, input_size:int = 40, batch_size:int=96) -> Any:
		list_batchs = []
		batch = []
		
		for i, image in enumerate(inputs, start=0):
			img_convert = self.preprocess(image, input_size)
			# print('IMG SHAPE = ', img_convert.shape)
			if i % batch_size != 0:
				batch.append(img_convert)
			else:
				batch.append(img_convert)
				list_batchs.append(np.array(batch))
				batch = []
		
		if batch != []:
			list_batchs.append(np.array(batch))

		return list_batchs