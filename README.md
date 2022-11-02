# Coupled Generative Adversarial Networks (CoGAN)
Coupled Generative Adversarial Networks, proposed by Liu et al [1] at NIPS 2016.
## Authors
Ming-Yu Liu, Oncel Tuzel
## Abstract 
We propose coupled generative adversarial network (CoGAN) for learning a joint distribution of multi-domain images. In contrast to the existing approaches, which require tuples of corresponding images in different domains in the training set, CoGAN can learn a joint distribution without any tuple of corresponding images. It can learn a joint distribution with just samples drawn from the marginal distributions. This is achieved by enforcing a weight-sharing constraint that limits the network capacity and favors a joint distribution solution over a product of marginal distributions one. We apply CoGAN to several joint distribution learning tasks, including learning a joint distribution of color and depth images, and learning a joint distribution of face images with different attributes. For each task it successfully learns the joint distribution without any tuple of corresponding images. We also demonstrate its applications to domain adaptation and image transformation.
<br />
[[Paper]](https://arxiv.org/abs/1606.07536)
<br />
<br />
In this repository, we utilized CoGAN with two version:
- COGAN-1
<br />

**Generative Models** 
Layer | Domain 1 | Domain 2 | Shared|
--- | --- | --- | ---|
1   | Conv2DTransposed(1024, K4x4, S1)-BN-PReLU | Conv2DTransposed(1024, K4x4, S1)-BN-PReLU | Yes|
2   | Conv2DTransposed(512, K3x3, S2)-BN-PReLU | Conv2DTransposed(512, K3x3, S2)-BN-PReLU  | Yes|
3   | Conv2DTransposed(256, K3x3, S2)-BN-PReLU | Conv2DTransposed(256, K3x3, S2)-BN-PReLU | Yes|
4   | Conv2DTransposed(128, K3x3, S2)-BN-PReLU | Conv2DTransposed(128, K3x3, S2)-BN-PReLU  | Yes|
5   | Conv2DTransposed(1, K6x6, S1)-**Tanh** | Conv2DTransposed(1, K6x6, S1)-**Tanh** | No|

**Discriminative Models** 
Layer | Domain 1 | Domain 2 | Shared|
--- | --- | --- | ---|
1   | Conv2D(20, K4x4, S1)-MAXPOOL(2) | Conv2D(20, K4x4, S1)-MAXPOOL(2) | No|
2   | Conv2D(50, K5x5, S1)-MAXPOOL(2) | Conv2D(50, K5x5, S1)-MAXPOOL(2) | Yes|
3   | FC(500)-PReLU | FC(500)-PReLU | Yes|
4   | FC(1)-**Sigmoid** | FC(1)-**Sigmoid** | Yes|
- COGAN-2
<br />

# Requirements
- Python 3.8
- Pytorch 1.12
- Cuda 11.6
- OpenCV
# How to train CoGAN
To train CoGAN, you can type the following command as follows:
```
$ python3 train.py --n_epochs 10 --lr 0.0002 --batch_size 32 --latent_dim 100 --d1 "org" --d2 "edge" --model "m1" --img_size 44
```
or 
```
$ sh run.sh
```

The configuration setting for training process: 
<br/>
- --**n_epochs** (int) : the number epochs for training process.  
- --**lr** (float) : the learning rate.
- --**batch_size** (int) : the batch size. 
- --**latent_dim** (int) : the latent dimension. 
- --**d1** (string): the domain 1 (org: original image, edge: edge image, rot: rotated image, neg: negative image). 
- --**d2** (string): the domain 2 (org: original image, edge: edge image, rot: rotated image, neg: negative image).
- --**model** (string): the selected models (m1 or m2). 
- --**img_size** (int): the image size.
# Results
1.**Generation of negative images using CoGAN**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/156000.png?raw=true)
<br />
2.**Generation of rotated images using CoGAN**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/rotate_m1.png?raw=true)
<br />
3.**Generation of edge images using CoGAN**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/edge_m1.png?raw=true)
# References
[1] [Ming-Yu Liu and Oncel Tuzel. Coupled generative adversarial networks. In D. Lee, M. Sugiyama,
U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing
Systems, volume 29. Curran Associates, Inc., 2016](https://arxiv.org/abs/1606.07536) 
<br />
[2] https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/
