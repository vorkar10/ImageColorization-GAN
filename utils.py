import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import cv2
import config
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset
import os
import glob

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
	print("=> Saving checkpoint")
	checkpoint = {
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict(),
	}
	torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
	print("=> Loading checkpoint")
	checkpoint = torch.load(checkpoint_file, map_location=config.device)
	model.load_state_dict(checkpoint["state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer"])

	# If we don't do this then it will just have learning rate of old checkpoint
	# and it will lead to many hours of debugging \:
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr


def read_image(image):
		
		image = np.array(image)
		width = image.shape[1]
		half_width = width // 2
	   
		input_image = image[:, :half_width, :]
		target_image = image[:, half_width:, :]
	
		input_image = input_image.astype(np.float32)
		target_image = target_image.astype(np.float32)
	
		return input_image, target_image

def crop_random(image, size):
	height, width, _ = size
	x, y = np.random.uniform(low=0,high=int(height-256)), np.random.uniform(low=0,high=int(width-256))  
	#return print(image.shape)
	return image[:, int(x):int(x)+256, int(y):int(y)+256]


def jittering_random(input_image, target_image, height=286, width=286):
	''' Crops image randomly, flips horizonatlly and image resizing'''
	# resizing to 286x286
	input_image = cv2.resize(input_image, (height, width) ,interpolation=cv2.INTER_NEAREST)
	target_image = cv2.resize(target_image, (height, width),
							   interpolation=cv2.INTER_NEAREST)

	stacked_image = np.stack([input_image, target_image], axis=0)
	cropped_image = crop_random(stacked_image, size=[config.IMG_HEIGHT, config.IMG_WIDTH, 3])
	
	input_image, target_image = cropped_image[0], cropped_image[1]
	#print(input_image.shape)
	if torch.rand(()) > 0.5:
	 # random mirroring
		input_image = np.fliplr(input_image)
		target_image = np.fliplr(target_image)
	return input_image, target_image
		
def normalize(input, target):
	''' Normalizes the input and target image'''
	input_image = (input / 127.5) - 1
	target_image = (target / 127.5) - 1
	return input_image, target_image
		
class Train_Normalize(object):
	def __call__(self, image):
		input, target = read_image(image)
		input, target = jittering_random(input, target)
		input, target = normalize(input, target)

		image_bw = torch.from_numpy(input.copy().transpose((2, 0, 1)))
		image_rgb = torch.from_numpy(target.copy().transpose((2, 0, 1)))
		return image_bw, image_rgb

	
class Val_Normalize(object):
	def __call__(self, image):
		input, target = read_image(image)

		input, target = normalize(input, target)

		image_bw = torch.from_numpy(input.copy().transpose((2,0,1)))
		image_rgb = torch.from_numpy(target.copy().transpose((2,0,1)))
		return image_bw, image_rgb

if __name__ == "__main__":
	from PIL import Image
	from torchvision.utils import save_image

	files = glob.glob("dataset/test/*/*.jpg")

	for fname in files:

		inp, trg = read_image(Image.open(fname))
		inp = (inp / 127.5) - 1
		image_bw = torch.from_numpy(input.copy().transpose((2,0,1)))

		save_image(torch.from_numpy(inp.transpose((2,0,1))), "x.png",normalize=True)
		save_image(torch.from_numpy(trg.transpose((2,0,1))), "y.png",normalize=True)

