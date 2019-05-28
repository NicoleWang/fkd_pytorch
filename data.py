import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

IMG_SIZE = 96
class FaceKeypointsDataset(Dataset):
    '''Face Keypoints Dataset'''
    def __init__(self, dataframe, train=True, transform=None):
        '''
        Args:
            dataframe (DataFrame): data in pandas dataframe format.
            train (Boolean) : True for train data with keypoints, default is True
            transform (callable, optional): Optional transform to be applied on 
            sample
        '''
        self.dataframe = dataframe
        self.train = train
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image = np.fromstring(self.dataframe.iloc[idx, -1], sep=' ')\
                .astype(np.float32).reshape(-1, IMG_SIZE)
        if self.train:
            keypoints = self.dataframe.iloc[idx, :-1].values.astype(np.float32)
        else:
            keypoints = None
        sample = {'image': image, 'keypoints': keypoints}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Normalize(object):
    '''Normalize input images'''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        return {'image': image / 255., # scale to [0, 1]
                'keypoints': keypoints}
        
class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.reshape(1, IMG_SIZE, IMG_SIZE)
        image = torch.from_numpy(image)
        if keypoints is not None:
            keypoints = torch.from_numpy(keypoints)
            return {'image': image, 'keypoints': keypoints}
        else:
            return {'image': image}

#class RandomHorizontalFlip(object):
#    '''
#    Horizontally flip image randomly with given probability
#    Args:
#        p (float): probability of the image being flipped.
#                   Defalut value = 0.5
#    '''
#    
#    def __init__(self, p=0.5):
#        self.p = p
#
#    def __call__(self, sample):
#        
#        flip_indices = [(0, 2), (1, 3),
#                        (4, 8), (5, 9), (6, 10), (7, 11),
#                        (12, 16), (13, 17), (14, 18), (15, 19),
#                        (22, 24), (23, 25)]
#        
#        image, keypoints = sample['image'], sample['keypoints']
#        
#        if np.random.random() < self.p:
#            image = image[:, ::-1]
#            if keypoints is not None:
#                for a, b in flip_indices:
#                    keypoints[a], keypoints[b]= keypoints[b], keypoints[a]
#                keypoints[::2] = 96. - keypoints[::2]
#        
#        return {'image': image, 
#                'keypoints': keypoints}
class RandomHorizontalFlip(object):
    '''
    Convert ndarrays in sample to Tensors.
    '''
    
    def __init__(self, p=0.5, dataset='A'):
        
        '''
        Args:
        p (float): probability of the image being flipped,
                   defalut value = 0.5
        dataset (string): A=All, L=dataset L, S=dataset S,
                   default value =  A
        '''
        self.p = p
        self.dataset = dataset

    def __call__(self, sample):
        
        if self.dataset == 'L':
            flip_indices = [(0, 2), (1, 3)]
        elif self.dataset == 'S':
            flip_indices = [(0, 4), (1, 5), (2, 6), (3, 7),
                            (8, 12), (9, 13), (10, 14), (11, 15),
                            (16, 18), (17, 19)]
        else:
            flip_indices = [(0, 2), (1, 3),
                            (4, 8), (5, 9), (6, 10), (7, 11),
                            (12, 16), (13, 17), (14, 18), (15, 19),
                            (22, 24), (23, 25)]
        
        image, keypoints = sample['image'], sample['keypoints']
        
        if np.random.random() < self.p:
            image = image[:, ::-1]
            if keypoints is not None:
                for a, b in flip_indices:
                    keypoints[a], keypoints[b]= keypoints[b], keypoints[a]
                keypoints[::2] = 96. - keypoints[::2]
        
        return {'image': image, 
                'keypoints': keypoints}
def prepare_train_valid_loaders(trainset, valid_size=0.2, 
                                batch_size=128):
    '''
    Split trainset data and prepare DataLoader for training and validation
    
    Args:
        trainset (Dataset): data 
        valid_size (float): validation size, defalut=0.2
        batch_size (int) : batch size, default=128
    ''' 
    
    # obtain training indices that will be used for validation
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=valid_sampler)
    
    return train_loader, valid_loader
