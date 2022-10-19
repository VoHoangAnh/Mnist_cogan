from importlib.metadata import requires
import sys 
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from torchvision import datasets  
from torch.utils.data import DataLoader 
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F

import argparse 
import os 
import numpy as np 

import mnistm  
from model import R_CoGANGenMNIST, R_CoGANDisMNIST
import matplotlib.pyplot as plt 
 
##############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help = "number of epochs of training")
parser.add_argument("--lr", type = float, default=0.0002, help = "Adam: learning rate")
parser.add_argument("--batch_size", type = int, default=32)
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help = "number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval betwen image samples")
opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False

##############################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
### Configure data loader ###
os.makedirs("../../data/mnist", exist_ok=True)
dataloader1 = torch.utils.data.DataLoader(
    mnistm.ONISTM(
        "../../data/mnistm",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

os.makedirs("../../data/mnistm", exist_ok=True)
dataloader2 = torch.utils.data.DataLoader(
    mnistm.MNISTM(
        "../../data/mnistm",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

#######################################
## Loss function
adversarial_loss = torch.nn.MSELoss()

## Initialize models
coupled_generators = R_CoGANGenMNIST()
coupled_discriminators = R_CoGANDisMNIST()

if cuda:
    coupled_generators.cuda()
    coupled_discriminators.cuda()
    
# Initialize weights
coupled_generators.apply(weights_init_normal)
coupled_discriminators.apply(weights_init_normal)

## Optimizers
optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ---------- 
print()
for epoch in range(opt.n_epochs):
    for i, ((imgs1, _), (imgs2, _)) in enumerate(zip(dataloader1, dataloader2)):

        batch_size = imgs1.shape[0]
        
        ## Aversarial ground truths 
        target = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
        
        ## Configure input 
        # print("input image")
        # print(imgs1.shape, imgs2.shape)
        imgs1 = Variable(imgs1.type(Tensor).expand(imgs1.size(0), 1, opt.img_size, opt.img_size))
        imgs2 = Variable(imgs2.type(Tensor).expand(imgs2.size(0), 1, opt.img_size, opt.img_size))
        #print(imgs1.shape, imgs2.shape)
        ##############################
        ## Train Generators 
        ##############################
        optimizer_G.zero_grad()
        
        ## Sample noise as generator input 
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        
        ## Generate a batch of images 
        gen_imgs1, gen_imgs2 = coupled_generators(z) 
        
        # print("generated image")
        # print(gen_imgs1.shape, gen_imgs2.shape)
        ## Determine target of generateed images 
        val1, val2 = coupled_discriminators(gen_imgs1, gen_imgs2)
        
        g_loss = (adversarial_loss(val1, target) + adversarial_loss(val2, target))/2
        
        g_loss.backward()
        optimizer_G.step()
        
        ##############################
        ## Train Discriminators 
        ##############################
        
        optimizer_D.zero_grad()
        
        ## Determine target of real and generated images 
        
        valid1_real, valid2_real = coupled_discriminators(imgs1, imgs2)
       
        valid1_fake, valid2_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())
        
        
        d_loss = (adversarial_loss(valid1_real, target) + adversarial_loss(valid1_fake, fake)
                  + adversarial_loss(valid2_real, target) + adversarial_loss(valid2_fake, fake) 
            )/4
        
        d_loss.backward()
        optimizer_D.step()
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader1), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader1) + i
        if batches_done % opt.sample_interval == 0:
            gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
            save_image(gen_imgs, "images/%d.png" % batches_done, nrow=8, normalize=True)