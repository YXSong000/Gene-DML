

import os
import yaml
from addict import Dict
import numpy as np
import pandas as pd
import torch

def match(x, y):
    """Returns a NumPy array of the positions of first occurrences of values in y in x. 

    Returns:
        np.array: a NumPy array of the positions of first occurrences of values in y in x
    """
    positions = np.empty_like(y, dtype=np.float64)
    for i, value in np.ndenumerate(y):
        try:
            positions[i] = np.where(x == value)[0][0]
        except IndexError:
            positions[i] = np.nan
    return positions.astype('int16')


def smooth_exp(cnt):
    """Apply smoothing to gene expression data in Pandas DataFrame.
    Take average gene expression of the nearest 9 spots.
    
    Args:
        cnt (pd.DataFrame): count data 

    Returns:
        pd.DataFrame: smoothed expression in DataFrame. 
    """

    ids = cnt.index
    delta = np.array([[1,0],
            [0,1],
            [-1,0],
            [0,-1],
            [1,1],
            [-1,-1],
            [1,-1],
            [-1,1],
            [0,0]])

    cnt_smooth = np.zeros_like(cnt).astype('float')

    for i in range(len(cnt)):
        spot = cnt.iloc[i,:]    
        
        # print(f"Smoothing {spot.name}")    
        center = np.array(spot.name.split('x')).astype('int')
        neighbors = center - delta
        neighbors = pd.DataFrame(neighbors).astype('str').apply(lambda x: "x".join(x), 1)
        
        cnt_smooth[i,:] = cnt[ids.isin(neighbors)].mean(0)
        
    cnt_smooth = pd.DataFrame(cnt_smooth)
    cnt_smooth.columns = cnt.columns
    cnt_smooth.index = cnt.index
    
    return cnt_smooth

def collate_fn(batch):
    """Custom collate function of train dataloader for TRIPLEX.   

    Args:
        batch (tuple): batch of returns from Dataset

    Returns:
        tuple: batch data
    """
    
    patch = torch.stack([item[0] for item in batch])
    exp = torch.stack([item[1] for item in batch])
    pid = torch.stack([item[2] for item in batch])
    sid = torch.stack([item[3] for item in batch])
    wsi = [item[4] for item in batch]
    position = [item[5] for item in batch]
    neighbors = torch.stack([item[6] for item in batch])
    mask = torch.stack([item[7] for item in batch])
    
    return patch, exp, pid, sid, wsi, position, neighbors, mask

# Load config
def load_config(config_name):
    """load config file in Dict

    Args:
        config_name (str): Name of config file. 

    Returns:
        Dict: Dict instance containing configuration.
    """
    config_path = os.path.join('./config', f'{config_name}.yaml')

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    
    return Dict(config)

