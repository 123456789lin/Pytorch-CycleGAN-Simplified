import os
from data.base_dataset import BaseDataset, get_transform_aligned
from data.image_folder import make_dataset
from PIL import Image
import glob

class AlignedDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A", '*')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B", '*')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(glob.glob(self.dir_A))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(glob.glob(self.dir_B))    # load images from '/path/to/data/trainB'
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform_A = get_transform_aligned(self.opt, grayscale=(opt.input_nc == 1))
        self.transform_B = get_transform_aligned(self.opt, grayscale=(opt.output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        
        A = A.float()
        B = B.float()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size
