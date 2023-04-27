import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchsummary import summary


from generator import Generator, generator_loss
from discriminator import Discriminator, discriminator_loss
import config
from utils import Train_Normalize, Val_Normalize
from common import get_norm_layer,weights_init
from utils import load_checkpoint, save_checkpoint
import os
from tqdm import tqdm
import pandas as pd
# os.environ['CUDA_VISIBLE_config.deviceS'] = '0'


def train(generator,discriminator, gen_optimizer, disc_optimizer, train_data_loader, validate_data_loader):

	start = 0
	df_loss = pd.DataFrame(columns=['Epoch',"D_loss", "G_loss"])

	if os.path.exists('model_loss_instancenorm.csv'):
		df_loss = pd.read_csv('model_loss_instancenorm.csv')
		start = int((max(df_loss['Epoch'].values)))
		print(start)


	disc_loss_plot, gen_loss_plot = [], []
	for epoch in range(start, config.num_epochs+1):


		disc_loss_list, gen_loss_list = [], []
		
		loop = tqdm(train_data_loader, leave=True)
		for (input_image, target_image),_ in loop:
		# for (input_image, target_image), _ in train_data_loader:
		   
			disc_optimizer.zero_grad()
			input_image = input_image.to(config.device)
			target_image = target_image.to(config.device)
			
			
			generated_image = generator(input_image)
			
			disc_input_generated = torch.cat((input_image, generated_image), 1)
			
			
			real_gt = Variable(torch.ones(input_image.size(0), 1, 30, 30).to(config.device))
			fake_gt = Variable(torch.zeros(input_image.size(0), 1, 30, 30).to(config.device))

			fake_pred = discriminator(disc_input_generated.detach())

			disc_fake_loss = discriminator_loss(fake_pred, fake_gt)

		
			disc_inp_real = torch.cat((input_image, target_image), 1)
			
											 
			output = discriminator(disc_inp_real)
			disc_real_loss = discriminator_loss(output, real_gt)

		  
			disc_total_loss = (disc_real_loss + disc_fake_loss) / 2
			disc_loss_list.append(disc_total_loss)
		  
			disc_total_loss.backward()
			disc_optimizer.step()
			
			
			# Train generator with real labels
			gen_optimizer.zero_grad()
			fake_gen = torch.cat((input_image, generated_image), 1)
			G = discriminator(fake_gen)
			gen_loss = generator_loss(generated_image, target_image, G, real_gt)
			gen_loss_list.append(gen_loss)

			gen_loss.backward()
			gen_optimizer.step()

			D_loss = torch.mean(torch.FloatTensor(disc_loss_list)).numpy()
			G_loss = torch.mean(torch.FloatTensor(gen_loss_list)).numpy()


			loop.set_postfix(
			Epoch=epoch,
			D_loss=D_loss,
			G_loss=G_loss
			)
			

			
		
		disc_loss_plot.append(torch.mean(torch.FloatTensor(disc_loss_list)))
		gen_loss_plot.append(torch.mean(torch.FloatTensor(gen_loss_list)))
		df_loss = df_loss.append({"Epoch":epoch,"D_loss":D_loss,"G_loss":G_loss},ignore_index=True)
		
		if epoch % 5 == 0:
			save_checkpoint(generator, gen_optimizer, filename=config.CHECKPOINT_GEN)
			save_checkpoint(discriminator, disc_optimizer, filename=config.CHECKPOINT_DISC)

			df_loss.to_csv("model_loss.csv",index=False)


		for (inputs, targets), _ in validate_data_loader:
			inputs = inputs.to(config.device)
			generated_output = generator(inputs)
			save_image(generated_output.data[:10], 'results/sample_%d'%epoch + '.png', nrow=5, normalize=True)



'''Data Preprocessing'''
def dataprep():
	train_data = ImageFolder(config.DIR, transform=transforms.Compose([
			Train_Normalize()]))
	train_data_loader = DataLoader(train_data, config.batch_size, shuffle=True)


	validation_data = ImageFolder(config.VAL_DIR, transform=transforms.Compose([
			Val_Normalize()]))
	validate_data_loader = DataLoader(validation_data, config.batch_size, shuffle=False)

	return train_data_loader, validate_data_loader



def main():

	train_data_loader, validate_data_loader = dataprep()

	if config.device == 'cuda':
			torch.cuda.device_count()

	# assigning the norm type
	normalization_layer = get_norm_layer(norm_type="instance")

	generator = Generator(3, 3, 64, norm_layer=normalization_layer)

	if config.LOAD_MODEL==False:
		print('Initializing weights for generator')
		generator.apply(weights_init)

	# For parallel data processing
	generator = torch.nn.DataParallel(generator)

	input = torch.ones(1, 3, 256, 256)
	#gen = generator(inp)

	input = input.to(config.device)
	generator = generator.to(config.device)

	print(summary(generator,(3,256,256)))


	discriminator = Discriminator(6, 64, norm_layer=normalization_layer)#.cuda().float()

	if config.LOAD_MODEL==False:
		print('Initializing weights for discriminator')
		discriminator.apply(weights_init)
	discriminator = torch.nn.DataParallel(discriminator)  # multi-GPUs

	input = torch.ones(1, 6, 256, 256)
	disc = discriminator(input)
	input = input.to(config.device)
	# discriminator = discriminator.to(config.device)

	# Print the model architecture
	summary(discriminator,(6,256,256))

	# Configuring the optimizer for both generator and discriminator
	gen_optimizer = optim.Adam(generator.parameters(), lr = config.learning_rate, betas=(0.5, 0.999))
	disc_optimizer = optim.Adam(discriminator.parameters(), lr = config.learning_rate, betas=(0.5, 0.999))


	# This was added to load the last model (in case the last training did not complete)
	if config.LOAD_MODEL:
		load_checkpoint(config.CHECKPOINT_GEN, generator, gen_optimizer, config.learning_rate)
		load_checkpoint(config.CHECKPOINT_DISC, discriminator, disc_optimizer, config.learning_rate)



	train(generator, discriminator, gen_optimizer, disc_optimizer, train_data_loader, validate_data_loader)


if __name__ == "__main__":
	main()