"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    """
    def __init__(self, opt):
        """Initialize the class; save the options in the class
        """
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        pass

def get_transform(opt, grayscale=False, method=Image.BICUBIC):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
        
        
    if opt.phase == "train":
        ##resize
        #osize = [286, 286] #scale images to this size
        #transform_list.append(transforms.Resize(osize, method))
        transform_list.append(transforms.RandomCrop(256))
        transform_list.append(transforms.RandomHorizontalFlip())
    elif opt.phase == "test":
        pass
    else:
        raise NotImplementedError('Parameter "opt.phase" can only be "train" or "test"')
        
        
    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
    return transforms.Compose(transform_list)
    
    
def get_transform_aligned(opt, grayscale=False, method=Image.BICUBIC):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    transform_list += [transforms.ToTensor()]
    
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
    return transforms.Compose(transform_list)

