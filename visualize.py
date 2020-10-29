############################# Import Section #################################

## Imports related to PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

## Generic imports
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

## Dependencies classes and functions
from utils import *
from H36Mseq_dloader import *

## Import Model
from SB_Dyan import SBDyan
from DyanOF import OFModel

############################# Import Section #################################


## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
iterations = 200
BATCH_SIZE = 8
LR = 0.001
gpu_id = 2
FRA = 50
PRE = 1
N_FRAME = FRA + PRE
N = NumOfPoles * 4
saveEvery = 20
tau = 0.05

## Load saved model
load_ckpt = False
checkptname = 'SBDyan'

actions = ["directions", "discussion", "eating", "greeting", "phoning",
           "posing", "purchases", "sitting", "sittingdown", "smoking",
           "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

dloader = H36Mseq(FRA, PRE, '/home/sandesh/data/dataset')


P, Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()


class Params:
    pass


args = Params()
args.input_dim = FRA
args.hid_dim1 = FRA
args.hid_dim2 = 2 * FRA
args.hid_dim3 = 2 * FRA
args.out_dim = 161
args.beta = torch.FloatTensor([1])
args.mode = 'sbl'

## Create the model
model_sb = SBDyan(Drr, Dtheta, FRA, PRE, gpu_id, args)
model_sb.cuda(gpu_id)

saved_sbl=torch.load('result_SBDyan-tau05/SBDyan300.pth')
model_sb.load_state_dict(saved_sbl['state_dict'])



model_fista = OFModel(Drr, Dtheta, FRA, PRE, gpu_id)
model_fista.cuda(gpu_id)
saved_fista=torch.load('result_Dyan_lam1/PoseDyan300.pth')
model_fista.load_state_dict(saved_fista['state_dict'])

if not os.path.isdir('Visual_plots'):
    os.mkdir('Visual_plots')

def plot_hist(c_sbl,c_fista):
    bins = np.linspace(-.7, .7, 30)
    plt.hist(c_sbl, bins, alpha=0.5, label='SBL')
    plt.hist(c_fista, bins, alpha=0.5, label='Fista')
    plt.legend(loc='upper right')
    plt.savefig('Visual_plots/HumanPose_sb_tau05.png')
    plt.close()


with torch.no_grad():
    data = dloader.get_train_batch(BATCH_SIZE)
    inputData = (data[:, 0:FRA, :]).cuda(gpu_id)
    expectedOut = data.cuda(gpu_id)
    start1 = time.time()
    y_hat, mu, _, alpha_loss = model_sb.forward(inputData)
    end1=time.time()
    print('Time taken for SBL model:', end1-start1)
    c_sbl=mu[0,:,:].flatten()
    start2 = time.time()
    c_fista=model_fista.forward2(inputData)
    end2 = time.time()
    c_fista = c_fista[0,:,:].flatten()
    print('Time taken for fista model:', end2 - start2)

    print('Sbl shape:', c_sbl.shape,'Fista shape:', c_fista.shape )

    plot_hist(c_sbl.data.cpu().numpy(),c_fista.data.cpu().numpy())


