from __future__ import print_function
from PIL import Image, ImageFilter
import cv2
import os
import numpy as np
import pickle as cPickle
import pickle
import gzip
import torch.utils.data as data
import urllib.request
from torchvision import datasets 

class MNISTM(data.Dataset):
    def __init__(self, root, domain1, domain2, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  ## training set or test set
        self.domains = [domain1, domain2]
        
        ## now load the picked numpy arrays
        if self.train:
            self.mnist_train = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_data
            self.mnist_train_labels = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_labels
        else:
            self.mnist_test = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_data
            self.mnist_test_labels = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_labels
            
    def __getitem__(self, index):
        if self.train:
            img, target = self.mnist_train[index], self.mnist_train_labels[index]
           
            img1 =  self.tranform_domain(img, self.domains[0])
            img2 =  self.tranform_domain(img, self.domains[1])
        else:
            return
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, target
    
    def tranform_domain(self, img, domain):
        ### domain: edge, neg, rot ###
        if domain=='org':
            img = Image.fromarray(img.squeeze().numpy())
        elif domain=='edge':
            img = self.edge(img)
        elif domain=='neg':
            img = self.inverse_image(img)
        elif domain=='rot':
            img = self.rotation(img)
            
        return img 
                    
                
    def rotation(self, img):
        img  = img.squeeze().numpy()
        (h, w) = img.shape[:2]
        (cx, cy) = (w//2, h//2)
        M = cv2.getRotationMatrix2D((cx, cy), 90, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        img = Image.fromarray(img)
        #img = img.rotate(90, expand = 1)
        return img
    
    def inverse_image(self, img):
        img = 255 - img
        img = Image.fromarray(img.squeeze().numpy())
        return img    
    
    def edge(self, img):
        img  = img.squeeze().numpy()
        img = cv2.Canny(img, 100, 200)
        img = Image.fromarray(img)
       
        return img 
    
    def __len__(self):
        if self.train:
            return 30000
        else:
            return 0
