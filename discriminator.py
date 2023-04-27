import torch
import torch.nn as nn
from common import adversarial_loss


def ConvBlock(in_channel, out_channel, norm_layer, stride):
	seq = [
			nn.Conv2d(in_channel, out_channel , kernel_size=4, stride=stride, padding=1, bias=False),
			norm_layer(out_channel),
			nn.LeakyReLU(0.2, True)
			]
	return seq


class Discriminator(nn.Module):
	"""PatchGAN discriminator"""

	def __init__(self, input_channels, filter_last=64, norm_layer=nn.BatchNorm2d):
		"""Construct a PatchGAN discriminator
		Parameters:
			input_channels  -- the number of channels in input images
			filter_last      -- the number of filters in the last conv layer
			layers -- the number of convolutional layers in the discriminator
			norm_layer      -- normalization layer
		"""
		super(Discriminator, self).__init__()
		kernel_size, padding, filter  = 4, 1, 1
		
		sequence = [nn.Conv2d(input_channels, filter_last, kernel_size=kernel_size, stride=2, padding=padding), nn.LeakyReLU(0.2, True)]
		
		filter_prev = 1
		for n in range(1, 3):  # gradually increase the number of filters
			filter_prev = filter
			filter = 2 ** n
			in_channel = filter_last * filter_prev
			out_channel = filter_last * filter
			sequence += ConvBlock(in_channel, out_channel, norm_layer, 2)

		filter_prev = filter
		filter = 8
		in_channel = filter_last * filter_prev
		out_channel = filter_last * filter
		sequence += ConvBlock(in_channel, out_channel, norm_layer, 1)

		sequence += [nn.Conv2d(filter_last * filter, 1, kernel_size=kernel_size, stride=1, padding=padding), nn.Sigmoid()]
		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		"""Standard forward."""
		return self.model(input)


def discriminator_loss(output, label):
	disc_loss = adversarial_loss(output, label)
	return disc_loss

