# Pytorch-CycleGAN-Simplified
Simplified the code of pytorch-cyclegan.

This code simplifies （ junyanz /pytorch-CycleGAN-and-pix2pix ，https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  ）

# dataset
Please refer to the data set acquisition: junyanz /pytorch-CycleGAN-and-pix2pix， CycleGAN train/test, Download a CycleGAN dataset 
# cycleGAN:
train:python train.py --dataroot ./datasets/dataset_name --name dataset_name_cyclegan --model cycle_gan
test:python test.py --dataroot ./datasets/dataset_name --name dataset_name_cyclegan --model cycle_gan
eg:
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan
