#!/bin/sh 
python3 train.py --n_epochs 200 --lr 0.0002 --batch_size 32 --latent_dim 100 --d1 "org" --d2 "neg" --model "m2" --img_size 28
