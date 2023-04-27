import torch
import torch.nn as nn
import torchvision
import functools


adversarial_loss = nn.BCELoss() 
l1_loss = nn.L1Loss()


# custom weights initialization called on generator and discriminator        
def weights_init(network, init_type='normal', scaling=0.02):
	'''Network weight initialization
	Parameters:
		network (network)   -- network to be initialized
		init_type (str) -- the name of an initialization method
		init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
	'''


	def init_func(m):
		'''define the initialization function'''
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
			torch.nn.init.normal_(m.weight.data, 0.0, scaling)
		elif classname.find('BatchNorm2d') != -1:
			torch.nn.init.normal_(m.weight.data, 1.0, scaling)
			torch.nn.init.constant_(m.bias.data, 0.0)


	network.apply(init_func)  # apply the initialization function <init_func>


def get_norm_layer(norm_type = 'batch'):
	"""Return a normalization layer (Batch/Instance)
	"""
	if norm_type == 'instance':
		normalization_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
	else:
		normalization_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
	return normalization_layer
