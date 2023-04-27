import torch
import torch.nn as nn
from common import adversarial_loss,l1_loss


def UpSampleConv(in_channel, out_channel, bias=True):
	return nn.ConvTranspose2d(in_channel, out_channel,
										kernel_size=4, stride=2,
										padding=1, bias=bias)

class Block(nn.Module):
	""" Block for Generator Unet with Skip connections
	"""

	def __init__(self, outer_channel, inner_channel, num_channels=None,
				 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
		""" Unet block with skip connections.
		Parameters:
			outer_channel  -- the number of channels in the outer conv layer
			inner_channel  -- the number of channels in the inner conv layer
			num_channels  -- the number of channels in input images/features
			submodule (UnetSkipConnectionBlock) -- previously defined submodules
			outermost    -- if this module is the outermost module
			innermost    -- if this module is the innermost module
			norm_layer    -- normalization layer
		"""
		super(Block, self).__init__()
		self.outermost = outermost
		if num_channels is None:
			num_channels = outer_channel
		# Downsampling layers
		downconv = nn.Conv2d(num_channels, inner_channel, kernel_size=4,
							 stride=2, padding=1, bias=False)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_channel)
		# Upsampling layers
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_channel)

		if outermost:
			in_channel, out_channel = inner_channel * 2, outer_channel
			upconv = UpSampleConv(in_channel, out_channel)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			in_channel, out_channel = inner_channel , outer_channel
			upconv =UpSampleConv(in_channel, out_channel, bias= False)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			in_channel, out_channel = inner_channel * 2, outer_channel
			upconv = UpSampleConv(in_channel, out_channel, bias= False)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]


			model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			# we can add skip connection
			return torch.cat([x, self.model(x)], 1)



class Generator(nn.Module):
	'''Unet generator'''

	def __init__(self, input_channels, output_channels, ch_last=64, norm_layer=nn.BatchNorm2d):
		'''Construct a Unet generator
		Parameters:
			input_channels (int)  -- the number of channels in input images
			output_channels (int) -- the number of channels in output images
			num_downs (int) -- the number of downsamplings in UNet
			ch_last (int)       -- the number of output channels/filter in the last conv layer
			norm_layer      -- normalization layer
		'''
		super(Generator, self).__init__()
		# construct unet structure
		# add the innermost layer
		block = Block(ch_last * 8, ch_last * 8, submodule=None, norm_layer=norm_layer, innermost=True)
		
		# add intermediate layers with ngf * 8 filters
		block = Block(ch_last * 8, ch_last * 8, submodule=block, norm_layer=norm_layer)
		block = Block(ch_last * 8, ch_last * 8, submodule=block, norm_layer=norm_layer)
		block = Block(ch_last * 8, ch_last * 8, submodule=block, norm_layer=norm_layer)
		
		# gradually reduce the number of filterss
		block = Block(ch_last * 4, ch_last * 8, submodule=block, norm_layer=norm_layer)
		block = Block(ch_last * 2, ch_last * 4, submodule=block, norm_layer=norm_layer)
		block = Block(ch_last, ch_last * 2, submodule=block, norm_layer=norm_layer)
		self.model = Block(output_channels, ch_last, num_channels=input_channels, submodule=block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

	def forward(self, input):
		"""Forward pass"""
		return self.model(input)


def generator_loss(generated_image, target_img, G, real_target):
	''' Calculating loss for generator'''
	gen_loss = adversarial_loss(G, real_target)
	l1_l = l1_loss(generated_image, target_img)
	gen_total_loss = gen_loss + (100 * l1_l)
	#print(gen_loss)
	return gen_total_loss