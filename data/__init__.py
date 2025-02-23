from importlib import import_module
import random

import torch.utils.data as data
from torch.utils import data as D
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

from data.srdata import SRData

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, SRData):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


# def create_dataset(opt):
#     """Create a dataset given the option.

#     This function wraps the class CustomDatasetDataLoader.
#         This is the main interface between this package and 'train.py'/'test.py'

#     Example:
#         >>> from data import create_dataset
#         >>> dataset = create_dataset(opt)
#     """
#     data_loader = CustomDatasetDataLoader(opt)
#     dataset = data_loader.load_data()
#     return dataset

def create_dataset(opt):
    dataloader = {}
    if opt.is_train:
        dataloader['train'], dataloader['validation'] = TrainDataloader(opt).get_datasets()
    else:
        dataloader['test'] = TestDataloader(opt).test_dataloader

    return dataloader


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.mode = datasets[0].mode

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class ABDataset(data.Dataset):
    def __init__(self, dataA, dataB, mode='unaligned'):
        self.dataA = dataA
        self.dataB = dataB
        self.sizeA = len(dataA)
        self.sizeB = len(dataB)
        self.mode = mode

        if self.mode == 'aligned':
            assert self.sizeA == self.sizeB, "number of dataset A and B should be equal"
        
        print('Number of samples in datasetsA:', self.sizeA)
        print('Number of samples in datasetsB:', self.sizeB)

    def __getitem__(self, idx):
        data_dictA = self.dataA[idx % self.sizeA]
        idxB = random.randint(0, self.sizeB - 1)
        data_dictB = self.dataB[idxB]


        return data_dictA, data_dictB
        
    def __len__(self):
        return max(self.sizeA, self.sizeB)


class TrainDataloader:
    def __init__(self, args):
        self.loader_train = None
        self.loader_train = None
        da_list = []
        db_list = []
        for dataset_name in args.dataA:
            print('dataset_name:', dataset_name)
            dataset_class = find_dataset_using_name(dataset_name)
            da_list.append(dataset_class(args, name=dataset_name))

        datasetsA = MyConcatDataset(da_list)

        if len(args.dataB) == 0:
            # datasets = datasetsA
            datasets = datasetsA
        else:
            for dataset_name in args.dataB:
                print('dataset_name:', dataset_name)
                dataset_class = find_dataset_using_name(dataset_name)
                db_list.append(dataset_class(args, name=dataset_name))

            datasetsB = MyConcatDataset(db_list)

            datasets = ABDataset(datasetsA, datasetsB, mode=args.dataset_mode)

        if len(args.dataVal) > 0:
            dval_list = []
            for dataset_name in args.dataVal:
                print('dataset_name:', dataset_name)
                dataset_class = find_dataset_using_name(dataset_name)
                dval_list.append(dataset_class(args, name=dataset_name, is_valid=True))
            train_d = datasets
            valid_d = MyConcatDataset(dval_list)
        else:
            valid_len = int(args.valid_ratio * len(datasets))
            train_len = len(datasets) - valid_len
            train_d, valid_d = D.random_split(datasets, lengths=[train_len, valid_len])

        print('Number of samples in {}:'.format(len(datasets)))
        print('Nunber of samples in training datasets:', len(train_d))
        print('Number of samples in validation datasets:', len(valid_d))

        # for i in range(len(train_datasets)):
        #     print("Number of train datasets: {}".format(len(train_datasets[i])))
        self.train_dataloader = DataLoader(
            # MyConcatDataset(train_datasets)
            train_d,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.n_threads,
        )
        self.valid_dataloader = DataLoader(
            valid_d,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.n_threads,
        )
    
    def get_datasets(self):
        return self.train_dataloader, self.valid_dataloader


class TestDataloader:
    def __init__(self, args):

        if args.test_dataset == '':
            print('test on {}'.format(args.dataA[0]))
            args.test_dataset = args.dataA[0]

        batch_size = 1

        dataset_name = args.test_dataset
        dataset_class = find_dataset_using_name(dataset_name)
        testset = dataset_class(args, name=dataset_name, is_train=False)
        if len(testset) != 0:
            self.test_dataloader = DataLoader(
                                        testset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=args.n_threads
                                    )
