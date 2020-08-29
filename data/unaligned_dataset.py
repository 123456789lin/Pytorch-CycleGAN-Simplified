import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import glob

class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A', '*')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B', '*')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(glob.glob(self.dir_A))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(glob.glob(self.dir_B))    # load images from '/path/to/data/trainB'
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform_A = get_transform(self.opt, grayscale=(opt.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(opt.output_nc == 1))

    def __getitem__(self, index):
    
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        
        #index_B = index % self.B_size
        index_B = random.randint(0, self.B_size - 1)# randomize the index for domain B to avoid fixed pairs.
        
        B_path = self.B_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
