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
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.url = '/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist.pkl.gz'
        self.filename_train_domain_1 = "mnist_train_original.pickle"
        self.filename_train_domain_2 = "mnist_train_edge.pickle"
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        #self.download()
        #self.create_two_domains()
        # now load the picked numpy arrays
        if self.train:
            self.mnist_train = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_data
            self.mnist_test = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_data
            self.mnist_train_labels = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_labels
            self.mnist_test_labels = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_labels

    def __getitem__(self, index):
        index_2 = np.random.randint(0, self.__len__(), 1)
        if self.train:
            img, target = self.mnist_train[index], self.mnist_train_labels[index]
            img = Image.fromarray(img.squeeze().numpy())
            img = img.rotate(90, expand = 1)
            
        else:
            return
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return 30000
        else:
            return 0

    def download(self):
        filename = os.path.join(self.root, self.filename)
        if os.path.isfile(filename):
            return
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print("Download %s to %s" % (self.url, filename))
        #urllib.urlretrieve(self.url, filename)
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

class ONISTM(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.url = '/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist.pkl.gz'
        self.filename_train_domain_1 = "mnist_train_original.pickle"
        self.filename_train_domain_2 = "mnist_train_edge.pickle"
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        #self.download()
        #self.create_two_domains()
        # now load the picked numpy arrays
        if self.train:
            self.mnist_train = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_data
            self.mnist_train_labels = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_labels
        else:
            self.mnist_test = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_data
            self.mnist_test_labels = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_labels 

    def __getitem__(self, index):
        index_2 = np.random.randint(0, self.__len__(), 1)
        if self.train:
            img, target = self.mnist_train[index], self.mnist_train_labels[index]
            img = Image.fromarray(img.squeeze().numpy())
           
        else:
            return
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return 30000
        else:
            return 0

    def download(self):
        filename = os.path.join(self.root, self.filename)
        if os.path.isfile(filename):
            return
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print("Download %s to %s" % (self.url, filename))
        #urllib.urlretrieve(self.url, filename)
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return
    
class MNISTM_Edge(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.url = '/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist.pkl.gz'
        self.filename_train_domain_1 = "mnist_train_original.pickle"
        self.filename_train_domain_2 = "mnist_train_edge.pickle"
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        #self.download()
        #self.create_two_domains()
        # now load the picked numpy arrays
        if self.train:
            self.mnist_train = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_data
            self.mnist_test = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_data
            self.mnist_train_labels = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_labels
            self.mnist_test_labels = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_labels

    def __getitem__(self, index):
        index_2 = np.random.randint(0, self.__len__(), 1)
        if self.train:
            img, target = self.mnist_train[index], self.mnist_train_labels[index]
            img = Image.fromarray(img.squeeze().numpy())
            # img = img.filter(ImageFilter.Kernel((3, 3), (-2, -2, -2, 0, 0,
            #                               0, 2, 2, 2), 1, 0))
            img = img.filter(ImageFilter.FIND_EDGES)
        else:
            return
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return 30000
        else:
            return 0
        

class MNISTM_NegImage(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.url = '/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist.pkl.gz'
        self.filename_train_domain_1 = "mnist_train_original.pickle"
        self.filename_train_domain_2 = "mnist_train_edge.pickle"
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        #self.download()
        #self.create_two_domains()
        # now load the picked numpy arrays
        if self.train:
            self.mnist_train = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_data
            self.mnist_test = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_data
            self.mnist_train_labels = datasets.MNIST(root="../../data/mnist", train=True, download=True).train_labels
            self.mnist_test_labels = datasets.MNIST(root="../../data/mnist", train=False, download=True).test_labels

    def __getitem__(self, index):
        index_2 = np.random.randint(0, self.__len__(), 1)
        if self.train:
            img, target = self.mnist_train[index], self.mnist_train_labels[index]
            img = 255 - img
            img = Image.fromarray(img.squeeze().numpy())
            
        else:
            return
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return 30000
        else:
            return 0