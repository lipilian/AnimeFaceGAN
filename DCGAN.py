#%% import package
import numpy as np
import torch as t 
import torchvision as tv 
import os 
import tqdm 
import matplotlib.pyplot as plt
import time
import random
import visdom
from torch import nn
# %%set the random seed for reproduction
manualSeed = 980
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
t.manual_seed(manualSeed);
# %% define hyperparameters
class Config(object):
    DataPath = './faces/'
    TestDataPath = './faces_codeTest/'
    # Workers = 0 for ipython kernel run
    # with python script we increase the workers number
    Workers = 0 # the number of processor used for data loading 
    ImgSize = 96
    BatchSize = 256
    NumChannels = 3 # color image has R G B channels
    LRG = 2e-4 # learning rate of generator
    LRD = 2e-4 # learning rate of discriminator
    Beta = 0.5 # Beta hyperparameters for Adam optimizer
    Gpu = True 
    NumGpu = 1 # the number of GPUs
    Nz = 100 # size of generator input
    Ngf = 64 # size of generator feature map
    Ndf = 64 # size of discriminator feature map
    SavePath = './Gfaces/' # generator saving path 

    Vis = True # if need to use visdom to visualize
    env = 'GAN' # visdom environment
    plot_every = 20 # visdom visualize every 20 batches

    DebugFile = 'tmp/debuggan' # start debug mode if this file exists
    DEvery = 1 # train discriminator every N batch
    GEvery = 5 # train generator every N batch
    SaveEvery = 10 # save model every N batch
    NetDPath = None # ./checkpoints/NetD_.pth pretrained discriminator model
    NetGPath = None # ./checkpoints/NetG_.pth pretrained generator model

    # test without training 
    GenImg = './result.png'
    # pick best 64 imgs from 512 imgs
    GenNum = 64
    GenSearch = 512 
    GenMean = 0 # mean of imgs
    GenStd = 1 # std of imgs
opt = Config()
# %% define dataset and dataloader 
transforms = tv.transforms.Compose([
    tv.transforms.Resize(opt.ImgSize),
    tv.transforms.CenterCrop(opt.ImgSize),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = tv.datasets.ImageFolder(opt.DataPath, transform = transforms)
# create the dataloader
dataloader = t.utils.data.DataLoader(
            dataset, 
            batch_size = opt.BatchSize, 
            shuffle=True,
            num_workers = opt.Workers,
            drop_last = True)
# %% plot some img for dataloader test 
device = t.device("cuda:0" if (t.cuda.is_available() and opt.NumGpu > 0) else "cpu")
test_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('training imgs')
plt.imshow(np.transpose(tv.utils.make_grid(test_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# %% custom the weigh initialization method called in NetG and NetD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# %% define the generator with classic method 
# TODO:  Replace 4 Layer Net with ResNet for faster training
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.NumGpu = opt.NumGpu
        self.main = nn.Sequential(
            # input vector z convert to 4 * 4 * 64 * 8 features first layer
            nn.ConvTranspose2d(opt.Nz, opt.Ngf * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(opt.Ngf * 8),
            nn.ReLU(True),
            # 8 * 8 * 64 * 4 second layer
            nn.ConvTranspose2d(opt.Ngf * 8, opt.Ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(opt.Ngf * 4),
            nn.ReLU(True),
            # 16 * 16  * 64 * 2 third layer 
            nn.ConvTranspose2d(opt.Ngf * 4, opt.Ngf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(opt.Ngf * 2),
            nn.ReLU(True),
            # 32 * 32 * 64 fourth layer
            nn.ConvTranspose2d(opt.Ngf * 2, opt.Ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(opt.Ngf),
            nn.ReLU(True),
            # 96 * 96 * 3 final image
            nn.ConvTranspose2d(opt.Ngf, 3, 5, 3, 1, bias = False),
            nn.Tanh() # output -1 to 1 value 
        )
    def forward(self, input):
        return self.main(input)
# %% define the discriminator with classic method
# TODO: Replace 4 Layer Net with ResNet for faster training 
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.NumGpu = opt.NumGpu
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(opt.NumChannels, opt.Ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.Ndf) x 32 x 32
            nn.Conv2d(opt.Ndf, opt.Ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.Ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.Ndf*2) x 16 x 16
            nn.Conv2d(opt.Ndf * 2, opt.Ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.Ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.Ndf*4) x 8 x 8
            nn.Conv2d(opt.Ndf * 4, opt.Ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.Ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.Ndf*8) x 4 x 4
            nn.Conv2d(opt.Ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # give the probability
        )
    def forward(self, input):
        return self.main(input).view(-1)
# %% start training
def train():



























# %%
'''
if __name__ == '__main__':
    startTime = time.time()
    test_batch = next(iter(dataloader))
    print("The time for dataloader with %d of workers: %f" % (opt.Workers, time.time() - startTime))
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('training imgs')
    plt.imshow(np.transpose(tv.utils.make_grid(test_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig('1.jpg')
'''
# %%
