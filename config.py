import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_WIDTH = 256
IMG_HEIGHT = 256

# device = 'cuda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16

DIR = './dataset/train'
VAL_DIR = './dataset/test'

CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

learning_rate = 2e-4
num_epochs = 10000

SAVE_MODEL = True
LOAD_MODEL = True

