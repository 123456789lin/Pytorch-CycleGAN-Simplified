"""This package includes all the modules related to data loading and preprocessing
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.unaligned_dataset import UnalignedDataset


def create_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        
        dataset_class = UnalignedDataset
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.unshuffle,
            num_workers=4)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
