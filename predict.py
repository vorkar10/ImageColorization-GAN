import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary


from generator import Generator, generator_loss
import config
from utils import Train_Normalize, Val_Normalize, read_image
from common import get_norm_layer,weights_init
from utils import load_checkpoint, save_checkpoint
import os
import pandas as pd
from tqdm import tqdm
import glob
import numpy as np
# os.environ['CUDA_VISIBLE_config.deviceS'] = '0'


if config.device == 'cuda':
		torch.cuda.device_count()

normalization_layer = get_norm_layer("instance")

generator = Generator(3, 3, 64, norm_layer=normalization_layer)#.cuda().float()



generator = torch.nn.DataParallel(generator)  # multi-GPUs

input = torch.ones(1, 3, config.IMG_HEIGHT, config.IMG_WIDTH)
#gen = generator(inp)

input = input.to(config.device)
generator = generator.to(config.device)

print(summary(generator,(3,config.IMG_HEIGHT,config.IMG_WIDTH)))

gen_optimizer = optim.Adam(generator.parameters(), lr = config.learning_rate, betas=(0.5, 0.999))


# if config.LOAD_MODEL:
load_checkpoint('Trained_Models/'+config.CHECKPOINT_GEN, generator, gen_optimizer, config.learning_rate)


from PIL import Image
from torchvision.utils import save_image

files = glob.glob("test_images/*.jpg")

for epoch, fname in enumerate(files):

	inp = np.array(Image.open(fname)).astype(np.float32)
	inp = (inp / 127.5) - 1
	image_bw = torch.from_numpy(inp.copy().transpose((2,0,1))).unsqueeze(0) 

	image_bw = image_bw.to(config.device)
	generated_output = generator(image_bw)
	save_image(generated_output.data[0], 'results/rgb_%d'%epoch + '.png', normalize=True)





