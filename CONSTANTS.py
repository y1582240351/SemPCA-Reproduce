import sys

sys.path.extend([".", ".."])
import os, gc, math, abc, pickle, argparse
import random
import time
import hashlib
import numpy as np
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from tqdm import *
import regex as re
import logging
from collections import Counter
from torch.nn.parameter import Parameter
from multiprocessing import Manager, Pool
from copy import deepcopy

print('Seeding everything...')
seed = 6
random.seed(seed)  # Python random module.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)  # Torch CPU random seed module.
torch.cuda.manual_seed(seed)  # Torch GPU random seed module.
torch.cuda.manual_seed_all(seed)  # Torch multi-GPU random seed module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONHASHSEED'] = str(seed)

print('Seeding Finished')

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SESSION = hashlib.md5(
    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 60 * 60)).encode('utf-8')).hexdigest()
SESSION = 'SESSION_' + SESSION


def GET_PROJECT_ROOT():
    # goto the root folder of LogBar
    current_abspath = os.path.abspath('__file__')
    while True:
        # change to logbar after testing
        if os.path.split(current_abspath)[1] == 'LogBar':
            project_root = current_abspath
            break
        else:
            current_abspath = os.path.dirname(current_abspath)
    return project_root


def GET_LOGS_ROOT():
    log_file_root = os.path.join(GET_PROJECT_ROOT(), 'logs')
    if not os.path.exists(log_file_root):
        os.makedirs(log_file_root)
    return log_file_root


LOG_ROOT = GET_LOGS_ROOT()
PROJECT_ROOT = GET_PROJECT_ROOT()
