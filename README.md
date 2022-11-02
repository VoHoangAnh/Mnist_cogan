# CoGAN
# Requirement
- Python3.8
- Pytorch 1.12
- Cuda11.6
# How to train CoGAN
```
$ python3 train.py --n_epochs 10 --lr 0.0002 --batch_size 32 --latent_dim 100 --d1 "org" --d2 "edge" --model "m2" --img_size 2828
```

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
Ming-Yu Liu and Oncel Tuzel. Coupled generative adversarial networks. In D. Lee, M. Sugiyama,
U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing
Systems, volume 29. Curran Associates, Inc., 2016
