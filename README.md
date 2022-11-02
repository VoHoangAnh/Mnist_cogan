# CoGAN
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
# Results
**Negative image**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/156000.png?raw=true)
<br />
**Rotated image**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/rotate_m1.png?raw=true)
<br />
**Edge image**
<br />
![alt text](https://github.com/VoHoangAnh/Mnist_cogan/blob/develop/mnistm/edge_m1.png?raw=true)
# Reference
[1] Ming-Yu Liu and Oncel Tuzel. Coupled generative adversarial networks. In D. Lee, M. Sugiyama,
U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing
Systems, volume 29. Curran Associates, Inc., 2016 
<br />
[2] https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/
