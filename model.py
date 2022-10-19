import torch 
import torch.nn as nn 

##### Transpose is learning parameter while Up-sampling is no-learning parameters. 
##### Using Up-samling for faster inference or 
##### training because it does not require to update weight or compute gradient

#########################
## CoGAN 
#########################
#####################
## Discriminator Net 
#####################
# class CoDisMNIST(nn.Module):
#     def __init__(self):
#         super(CoDisMNIST, self).__init__()
#         ## Domain A ##
#         self.conv0_a = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
#         self.pool0_a = nn.MaxPool2d(kernel_size=2) 
#         ## Domain B ##
#         self.conv0_b = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
#         self.pool0_b = nn.MaxPool2d(kernel_size=2)
#         ## Shared convolution ##
#         self.D = nn.Sequential(
#             nn.Linear(8000, 500),
#             #nn.Linear(3920, 500),
#             nn.PReLU(),
#             nn.Linear(500,1),
#             nn.Sigmoid()
#         ) 
#         #self.sigmoid = nn.Sigmoid()
        
#     def forward(self, xa, xb):
#         h0_a = self.pool0_a(self.conv0_a(xa))
#         h0_b = self.pool0_b(self.conv0_b(xb))
#         out_a = h0_a.view(h0_a.shape[0], -1)
#         #print(out_a.shape)
#         #print(h0_a.shape)
#         out_a = self.D(out_a)
        
#         #out_a = self.sigmoid(out_a)
#         out_b = h0_a.view(h0_b.shape[0], -1)
#         out_b = self.D(out_b)
#         #out_b = self.sigmoid(out_b)
        
#         return out_a, out_b

class CoDisMNIST(nn.Module): ## shared_weights
    def __init__(self):
        super(CoDisMNIST, self).__init__()
        ## Domain A ##
        self.conv0_a = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.pool0_a = nn.MaxPool2d(kernel_size=2) 
        ## Domain B ##
        self.conv0_b = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.pool0_b = nn.MaxPool2d(kernel_size=2)
        
        self.shared_weight = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2)
        )
        ## Shared convolution ##
        self.D = nn.Sequential(
            nn.Linear(3200, 500),
            nn.PReLU(),
            nn.Linear(500,1),
            nn.Sigmoid()
        ) 
        
    def forward(self, xa, xb):
        h0_a = self.pool0_a(self.conv0_a(xa))
        h0_b = self.pool0_b(self.conv0_b(xb))
       
        o = torch.cat((h0_a, h0_b), 0)
        h = self.shared_weight(o)
        out_ = h.view(h.shape[0], -1)
        out = self.D(out_)
        
        h_a = out[:xa.shape[0]]
        h_b = out[xa.shape[0]:]
        
        return h_a, h_b
    
###################    
## Generator Net ##
###################
class CoGenMNIST(nn.Module):
    def __init__(self):
        super(CoGenMNIST, self).__init__()
        latent_dims = 100
        self.shared_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dims, out_channels=1024, kernel_size=4, stride = 1), 
            nn.BatchNorm2d(1024), 
            nn.PReLU(),
            
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride = 2),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride = 2),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride = 2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
       
        self.G1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=6, stride = 1),
            nn.BatchNorm2d(1,0.8),
            nn.PReLU()
            )
        
        self.G2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=6, stride = 1),
            nn.BatchNorm2d(1,0.8),
            nn.PReLU()
            )
        self.tanh = nn.Tanh()               
        
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.shared_conv(z)
        out_a = self.tanh(self.G1(h))
        out_b = self.tanh(self.G2(h))
        
        return out_a, out_b
   
##################################################
## CoGAN - rotation 
##################################################
class R_CoGANGenMNIST(nn.Module):
    def __init__(self):
        super(R_CoGANGenMNIST, self).__init__()
        
        self.sw = nn.Sequential(
            nn.Linear(100, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
             
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
             
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
    
        )
        self.G1 = nn.Sequential(
            nn.Linear(1024, 784),
            nn.Tanh()
        )
        
        self.G2 = nn.Sequential(
            nn.Linear(1024, 784),
            nn.Tanh()
        )
        
    def forward(self, z):    
        z = z.view(z.size(0), z.size(1))        
        h = self.sw(z)
        out_a = self.G1(h)
        out_b = self.G2(h)
        
        return out_a.view(z.size(0), 1, 28, 28), out_b.view(z.size(0), 1, 28, 28)

class R_CoGANDisMNIST(nn.Module):
    def __init__(self):
        super(R_CoGANDisMNIST, self).__init__()
        ## Domain A ##
        self.conv0_a = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.pool0_a = nn.MaxPool2d(kernel_size=2) 
        self.conv1_a = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.pool1_a = nn.MaxPool2d(kernel_size=2) 
        
        ## Domain B ##
        self.conv0_b = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.pool0_b = nn.MaxPool2d(kernel_size=2)
        self.conv1_b = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.pool1_b = nn.MaxPool2d(kernel_size=2)
        
        self.shared_weight = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc = nn.Sequential(
            #nn.Linear(2880, 500),
            nn.Linear(800, 500),
            nn.PReLU(),
        )
        ## Shared convolution ##
        self.D = nn.Sequential(
            #nn.Linear(2880, 500),
            #nn.Linear(800, 500),
            #nn.PReLU(),
            #nn.Dropout(0.3),
            nn.Linear(500,1),
            nn.Sigmoid()
        ) 
        
    def forward(self, xa, xb):
       
        h0_a = self.pool0_a(self.conv0_a(xa))
        #h1_a = self.pool1_a(self.conv1_a(h0_a))
        
        h0_b = self.pool0_b(self.conv0_b(xb))
        #h1_b = self.pool1_b(self.conv1_b(h0_b))
        
        #out_a = h0_a.view(h1_a.shape[0], -1)
        #out_b = h0_a.view(h1_b.shape[0], -1)
       
        
        o = torch.cat((h0_a, h0_b), 0)
        h = self.shared_weight(o)
        out_ = h.view(h.shape[0], -1)
        #print(out_.shape)
        out_a = self.fc(out_[:32])
        out_b = self.fc(out_[32:])
        out_ = torch.cat((out_a, out_b), 0)
        out = self.D(out_)
        out_a = out[:xa.shape[0]]
        out_b = out[xa.shape[0]:]
        return out_a, out_b
        
        
##################################################
## Conditional GAN
##################################################
