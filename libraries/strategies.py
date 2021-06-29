import cv2  

import pickle 
import numpy as np 
import operator as op 
import itertools as it, functools as ft 


import torch as th
import torchvision as tv 

from os import path 
from glob import glob 
from torchvision import transforms as T 


def pull_files(target, rule):
	return glob(path.join(target, rule))

def read_image(image_path, by='cv'):
	if by == 'cv':
		return cv2.imread(image_path, cv2.IMREAD_COLOR)
	if by == 'th':
		return tv.io.read_image(image_path)
	raise ValueError(by)

def th2cv(tensor_3d):
	red, green, blue = tensor_3d.numpy()
	return cv2.merge((blue, green, red))

def cv2th(bgr_image):
	blue, green, red = cv2.split(bgr_image)
	return th.from_numpy(np.stack([red, green, blue]))

def to_grid(batch_images, nb_rows=8, padding=10, normalize=True):
	grid_images = tv.utils.make_grid(batch_images, nrow=nb_rows, padding=padding, normalize=normalize)
	return grid_images
