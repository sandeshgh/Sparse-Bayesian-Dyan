############################# Import Section #################################
####Sparse Bayesian DYAN######
#Author: Sandesh Ghimire
#This code is built upon the base code due to liuem607 (Wenqian Liu)
#https://github.com/liuem607/DYAN

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
#import matplotlib.pyplot as plt

## Dependencies classes and functions
from utils import *
from H36Mseq_dloader import *

## Import Model
from SB_Dyan import SBDyan

############################# Import Section #################################



## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
iterations = 200
BATCH_SIZE = 1
LR = 0.001
gpu_id = 2
FRA = 50 
PRE = 1
N_FRAME = FRA+PRE
N = NumOfPoles*4
saveEvery = 20
tau = 0.05


## Load saved model 
load_ckpt = False
checkptname = 'SBDyan'

actions = ["directions", "discussion", "eating", "greeting", "phoning",
			"posing", "purchases", "sitting", "sittingdown", "smoking",
			"takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

dloader = H36Mseq(FRA, PRE, '/home/sandesh/data/dataset')

## Load input data
# set train list name:
# trainFolderFile = 'trainlist01.txt'
# set training data directory:
# rootDir ='/mnt/Data/wen/Kitti_Flows/'# '/data/Abhishek/UCF_Flows/'
#trainFoldeList = getListOfFolders(trainFolderFile)[::10]
# if Kitti dataset: use listOfFolders instead of trainFoldeList
# listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]


# trainingData = videoDataset(folderList=listOfFolders,
# 							rootDir=rootDir,
# 							N_FRAME=N_FRAME)

# dataloader = DataLoader(trainingData, 
# 						batch_size=BATCH_SIZE ,
# 						shuffle=True, num_workers=1)

## Initializing r, theta
P, Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()


class Params:
	pass

# Parameters accumulated in the structure Params to be passed to models etc.
args = Params()
args.input_dim = FRA
args.hid_dim1 = FRA
args.hid_dim2 = 2*FRA
args.hid_dim3 = 2*FRA
args.out_dim = 161
args.beta=torch.FloatTensor([1])
args.mode='sbl'

## Create the model
model = SBDyan(Drr, Dtheta, FRA, PRE, gpu_id, args)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1) # if Kitti: milestones=[100,150]
loss_mse = nn.MSELoss()
start_epoch = 1

## If want to continue training from a checkpoint
if(load_ckpt):
	loadedcheckpoint = torch.load(ckpt_file)
	start_epoch = loadedcheckpoint['epoch']
	model.load_state_dict(loadedcheckpoint['state_dict'])
	optimizer.load_state_dict(loadedcheckpoint['optimizer'])
	

print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()

## Start the Training
for epoch in range(start_epoch, EPOCH+1):
	start = time.time()
	loss_value = []
	reconstruct_loss=[]
	scheduler.step()
	for iter in range(iterations):
		data = dloader.get_train_batch(BATCH_SIZE)
		inputData = Variable(data[:,0:FRA,:]).cuda(gpu_id)
		expectedOut = Variable(data).cuda(gpu_id)
		optimizer.zero_grad()
		y_hat, _, _, alpha_loss = model.forward(inputData)
		rec_loss = loss_mse(y_hat, expectedOut)
		loss = rec_loss+tau*alpha_loss
		loss.backward()
		optimizer.step()
		loss_value.append(loss.data.item())
		reconstruct_loss.append(rec_loss.data.item())
	# for i_batch, sample in enumerate(dloader):
	# 	print(sample.shape)
	# 	data = sample['frames'].squeeze(0).cuda(gpu_id)
	# 	expectedOut = Variable(data)
	# 	inputData = Variable(data[:,0:FRA,:])
	# 	optimizer.zero_grad()
	# 	output = model.forward(inputData)
	# 	loss = loss_mse(output[:,FRA], expectedOut[:,FRA]) # if Kitti: loss = loss_mse(output, expectedOut)
	# 	loss.backward()
	# 	optimizer.step()
	# 	loss_value.append(loss.data.item())

	loss_val = np.mean(np.array(loss_value))
	reconstruct_loss_val = np.mean(np.array(reconstruct_loss))
	end_t=time.time()
	print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| reconstruct loss: %.4f' % reconstruct_loss_val)
	print('Time:', end_t-start)

	if not os.path.isdir('result_SBDyan-tau05'):
		os.mkdir('result_SBDyan-tau05')
	if epoch % saveEvery ==0 :
		save_checkpoint({	'epoch': epoch + 1,
							'state_dict': model.state_dict(),
							'optimizer' : optimizer.state_dict(),
							},'result_SBDyan-tau05/'+checkptname+str(epoch)+'.pth')
