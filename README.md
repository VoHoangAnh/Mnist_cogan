# Coupled Generative Adversarial Networks (CoGAN)
Coupled Generative Adversarial Networks was proposed by Liu et al [1] at NIPS 2016.
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

# Requirements
- Python 3.8
- Pytorch 1.12
- Cuda 11.6
# How to train CoGAN
To train CoGAN, you can type the following command as follows:
```
$ python3 train.py --n_epochs 10 --lr 0.0002 --batch_size 32 --latent_dim 100 --d1 "org" --d2 "edge" --model "m2" --img_size 28
```
or 
```
$ sh run.sh
```

The cofiguration setting: 
<br/>
- --n_epochs 200 
- --lr 0.0002 
- --batch_size 32 
- --latent_dim 100 
- --d1 "rot" 
- --d2 "edge" 
- --model "m1" 
- --img_size 44
# Results
**Negative images**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/156000.png?raw=true)
<br />
**Rotated images**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/rotate_m1.png?raw=true)
<br />
**Edge images**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/edge_m1.png?raw=true)
# References
[1] Ming-Yu Liu and Oncel Tuzel. Coupled generative adversarial networks. In D. Lee, M. Sugiyama,
U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing
Systems, volume 29. Curran Associates, Inc., 2016 
<br />
[2] https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/
