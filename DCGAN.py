#%% import package
import numpy as np
import ipdb
import glob
import torch as t 
import torchvision as tv 
import os 
import tqdm 
import matplotlib.pyplot as plt
import time
import random
import visdom
from visdom import server
from torchnet.meter import AverageValueMeter
# in order to use visdom via ssh, set ssh config forawrd
# 127.0.0.1:8097 to local 127.0.0.1:8097
from torch import nn
from visualize import Visualizer
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
    MaxEpochs = 400
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
    checkpointsPath ='./checkpoints/'

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
opt.Vis = True # set the visulization to true by using visdom
# run python -m visdom.server in terminal
if opt.Vis:
    vis = Visualizer(opt.env)
# create generator and discriminator
netG, netD = Generator(opt), Discriminator(opt)
# load pretrained check points
map_location = lambda storage, loc: storage
netDCheckPath = 
if opt.NetDPath:
    netD.load_state_dict(t.load(opt.NetDPath, map_location=map_location))
if opt.NetGPath:
    netG.load_state_dict(t.load(opt.NetGPath, map_location=map_location))
netG.to(device)
netD.to(device)
netG.apply(weights_init);
netD.apply(weights_init);
# %% define lost and optimizer
criterion = t.nn.BCELoss().to(device)
optimizerG = t.optim.Adam(netG.parameters(), lr = opt.LRG, betas=(opt.Beta, 0.999))
optimizerD = t.optim.Adam(netD.parameters(), lr = opt.LRD, betas=(opt.Beta, 0.999))
# define true label and fake label 
TrueLabel = t.ones(opt.BatchSize).to(device)
FakeLabel = t.zeros(opt.BatchSize).to(device)
fix_noises = t.randn(opt.BatchSize, opt.Nz, 1, 1).to(device)
errorGMeter = AverageValueMeter()
errorDMeter = AverageValueMeter()
# %% start training
print('start training ......')
for epoch in range(opt.MaxEpochs):
    for i, (img,_) in tqdm.tqdm(enumerate(dataloader)):
        realImg = img.to(device) # transfer 256,3,96,96 image to CUDA
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        if i % opt.DEvery == 0: # train discriminator
            #maximize log(D(x)) + log(1 - D(G(z)))
            optimizerD.zero_grad()
            # train all real image with true label
            output = netD(realImg).view(-1)
            errorDReal = criterion(output, TrueLabel)
            errorDReal.backward()
            # train all fake image with fake label
            noises = t.randn(opt.BatchSize, opt.Nz, 1, 1, device = device)
            fakeImg = netG(noises) 
            output = netD(fakeImg.detach()).view(-1)# turn off gradient tracking to avoid generator training
            errorDFake = criterion(output, FakeLabel)
            errorDFake.backward()
            # update all weight
            optimizerD.step()
            # collect all errors
            errorD = errorDReal + errorDFake
            errorDMeter.add(errorD.item())
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if i % opt.GEvery == 0: # train generator 
            optimizerG.zero_grad()
            if i % opt.DEvery != 0: # if discriminator is not trained within this batch
                noises = t.randn(opt.BatchSize, opt.Nz, 1, 1, device = device)
                fakeImg = netG(noises)
            # if discriminator got trained, we already have trained fakeImg
            output = netD(fakeImg).view(-1)
            errorG = criterion(output, TrueLabel)
            errorG.backward()
            optimizerG.step()
            errorGMeter.add(errorG.item())
        ############################
        # (3) visualize with visdom
        ###########################  
        if opt.Vis and i % opt.plot_every == opt.plot_every - 1:
            if os.path.exists(opt.DebugFile):
                ipdb.set_trace()
            fixFakeImg = netG(fix_noises)
            vis.images(fixFakeImg.detach().cpu().numpy()[:64] * 0.5 + 0.5, win = 'fixFakeImg')
            vis.images(realImg.data.cpu().numpy()[:64] * 0.5 + 0.5, win = 'realImg')
            vis.plot('ErrorGenerator', errorGMeter.value()[0])
            vis.plot('ErrorDiscriminator', errorDMeter.value()[0])
    ############################
    # (3) save training result
    ###########################        
    if (epoch + 1) % opt.SaveEvery == 0: 
        tv.utils.save_image(fixFakeImg.data[:64], '%s/%05d.png' % (opt.SavePath, epoch), normalize=True, range = (-1,1))
        t.save(netD.state_dict(), './checkpoints/netD_%05d.pth' % epoch) 
        t.save(netG.state_dict(), './checkpoints/netG_%05d.pth' % epoch)      
        errorDMeter.reset()
        errorGMeter.reset()
# %%
t.save(netD.state_dict(), './checkpoints/netD_%05d.pth' % 9) 
# %%
