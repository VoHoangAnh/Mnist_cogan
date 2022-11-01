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
import time 
import mnistm 
from model import CoDisMNIST, CoGenMNIST, R_CoGANGenMNIST, R_CoGANDisMNIST
import matplotlib.pyplot as plt 
 
##############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help = "number of epochs of training")
parser.add_argument("--lr", type = float, default=0.0002, help = "Adam: learning rate")
parser.add_argument("--batch_size", type = int, default=32)
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--d1", default="org", help="domain 1")
parser.add_argument("--d2", default="edge", help="domain 2")
parser.add_argument("--model", default="m2", help="the architecture of CoGAN")
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
## starting time 
start = time.time()
dataloader = torch.utils.data.DataLoader(
    mnistm.MNISTM(
        "../../data/mnist",
        opt.d1,
        opt.d2,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
### end time 
end = time.time()
print(f"Runtime of the dataloader is {end - start}")     
#######################################
## Loss function
adversarial_loss = torch.nn.MSELoss()

## Initialize models: 44
if opt.model=="m1":
    coupled_generators = CoGenMNIST(opt.latent_dim)
    coupled_discriminators = CoDisMNIST()
elif opt.model=="m2":
## Initialize models: 28
    coupled_generators = R_CoGANGenMNIST()
    coupled_discriminators = R_CoGANDisMNIST()

if cuda:
    coupled_generators.cuda()
    coupled_discriminators.cuda()
    
## Initialize weights
coupled_generators.apply(weights_init_normal)
coupled_discriminators.apply(weights_init_normal)

## Optimizers
optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ---------- 

train_loss_d = []
train_loss_g = []
for epoch in range(opt.n_epochs):
    loss_d_value = 0.0
    loss_g_value = 0.0
    for i, (imgs1, imgs2, _) in enumerate(dataloader):

        batch_size = imgs1.shape[0]
        
        ## Aversarial ground truths 
        target = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
        
        ## Configure input 
        # print("input image")
        # print(imgs1.shape, imgs2.shape)
        imgs1 = Variable(imgs1.type(Tensor).expand(imgs1.size(0), 1, opt.img_size, opt.img_size))
        imgs2 = Variable(imgs2.type(Tensor).expand(imgs2.size(0), 1, opt.img_size, opt.img_size))
        
        ##############################
        ## Train Generators 
        ##############################
        optimizer_G.zero_grad()
        
        ## Sample noise as generator input 
        z = Variable(Tensor(np.random.normal(0,1, (batch_size, opt.latent_dim))))
        
        ## Generate a batch of images 
        gen_imgs1, gen_imgs2 = coupled_generators(z) 
        
        # print("generated image")
        # print(gen_imgs1.shape, gen_imgs2.shape)
        ## Determine target of generateed images 
        val1, val2 = coupled_discriminators(gen_imgs1, gen_imgs2)  
        
        g_loss = (adversarial_loss(val1, target) + adversarial_loss(val2, target))/2 ## MSE loss 
        
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
        
        loss_d_value+=d_loss.item()
        loss_g_value+=g_loss.item()
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
            save_image(gen_imgs, "images/%d.png" % batches_done, nrow=8, normalize=True)
    
    train_loss_d.append(loss_d_value/len(dataloader))
    train_loss_g.append(loss_g_value/len(dataloader))

##########################
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# plt.plot(train_loss_d, label = "discriminator loss")
# plt.plot(train_loss_g, label = "generator loss")
# plt.xlabel("Iterations")
# plt.ylabel("Loss value")
# plt.legend(["discriminator loss", "generator loss"])
# plt.savefig("./rotate_loss.pdf", format='pdf')