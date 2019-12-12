import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler, SubsetRandomSampler

def split_train_val(train_size, network_input, network_output, batch_size):
    '''
    Divide the dataset into train/val

    Parameters:
        - 

    Returns:
        - 

    '''
    indices = list(range(len(network_input)))
    split = int(np.floor(train_size*len(network_input)))
    train_idx, val_idx = indices[:split], indices[split:]

    train_sampler = SequentialSampler(train_idx)

    val_sampler = SequentialSampler(val_idx)

    dataset = TensorDataset(network_input,network_output)
    train_loader = DataLoader(dataset, batch_size= batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size= batch_size,sampler= val_sampler)

    return train_loader, val_loader

