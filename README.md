# Pytorch-CycleGAN-Simplified
Simplified the code of pytorch-cyclegan.

This code simplifies （ junyanz /pytorch-CycleGAN-and-pix2pix ，https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  ）

# dataset

Please refer to the data set acquisition: junyanz /pytorch-CycleGAN-and-pix2pix， CycleGAN train/test, Download a CycleGAN dataset 

Data set structure:（reference horse2zebra ）


├── datasets                   
|       ├── <dataset_name>      # i.e. horse2zebra       
|       |       ├── trainA             
|       |       ├── trainB             
|       |       ├── testA         
|       |       ├── testB            
             

# cycleGAN:

train:python train.py --dataroot ./datasets/dataset_name --name dataset_name_cyclegan --model cycle_gan

test:python test.py --dataroot ./datasets/dataset_name --name dataset_name_cyclegan --model cycle_gan

Data can be paired or unpaired

eg:

python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan

python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan

# pix2pix:

train:python train.py --dataroot ./datasets/dataset_name --name pix2pix_dataset_name --model pix2pix

test:python test.py --dataroot ./datasets/dataset_name --name pix2pix_dataset_name --model pix2pix

Data must be paired


