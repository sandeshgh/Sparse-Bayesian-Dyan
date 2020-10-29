####Sparse Bayesian DYAN######
#Author: Sandesh Ghimire
#This code is built upon the base code due to liuem607 (Wenqian Liu)
#https://github.com/liuem607/DYAN

############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

from math import sqrt
import numpy as np


############################# Import Section #################################

# Create Dictionary
def creatRealDictionary(T, Drr, Dtheta, gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones = Variable(Wones, requires_grad=False)
    for i in range(0, T):
        W1 = torch.mul(torch.pow(Drr, i), torch.cos(i * Dtheta))
        W2 = torch.mul(torch.pow(-Drr, i), torch.cos(i * Dtheta))
        W3 = torch.mul(torch.pow(Drr, i), torch.sin(i * Dtheta))
        W4 = torch.mul(torch.pow(-Drr, i), torch.sin(i * Dtheta))
        W = torch.cat((Wones, W1, W2, W3, W4), 0)

        WVar.append(W.view(1, -1))
    dic = torch.cat((WVar), 0)
    G = torch.norm(dic, p=2, dim=0)
    idx = (G == 0).nonzero()
    nG = G.clone()
    # print(idx)
    nG[idx] = np.sqrt(T)
    G = nG

    dic = dic / G

    return dic



class Alpha_net(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, hid_dim3, out_dim):
        super(Alpha_net, self).__init__()
        self.out_dim=out_dim
        self.l1 = nn.LSTM(input_size=input_dim, hidden_size=hid_dim1, num_layers=2, batch_first=True, bidirectional=True)
        self.l2 = nn.LSTM(input_size=2*hid_dim1, hidden_size=hid_dim2, num_layers=2, batch_first=True, bidirectional=True)
        self.l3 = nn.LSTM(input_size=2*hid_dim2, hidden_size=out_dim, num_layers=2, batch_first=True, bidirectional=True)
        # self.fc=nn.Linear(hid_dim3,1)
        # self.act=nn.ReLU()
        # self.bl=nn.Sequential(self.l1, self.l2)

    def forward(self, x):
        # alpha=(self.bl(x))
        x=x.permute(0, 2, 1)
        out = self.l1(x)
        out=self.l2(out[0])
        out=self.l3(out[0])[0]
        alpha = 0.5*(out[:,:,:self.out_dim] + out[:,:,self.out_dim:])
        alpha= alpha**2
        return alpha.permute(0, 2, 1)



class Encoder(nn.Module):
    def __init__(self, Drr, Dtheta, T, gpu_id, beta, args):
        #mode = sbl means sparse bayesian learning mode.
        #mode = prior_learning means prior is itself learnt to optimize the future reconstructions
        super(Encoder, self).__init__()
        self.mode=args.mode
        self.rr = Drr
        self.theta = Dtheta
        self.T = T
        self.I= torch.diag(torch.ones(T)).cuda(gpu_id)
        self.gid = gpu_id
        self.beta=beta
        self.alpha_net= Alpha_net(args.input_dim, args.hid_dim1, args.hid_dim2, args.hid_dim3, args.out_dim)

    def compute_posterior(self, D, var,y):
        # y ~ T, D~T*160, var ~160*pixels
        #Computes the Bayesian posterior in the batch mode
        #Assumes D is 2dim, y is B*T*P and y is a
        beta=self.beta
        # y=y.permute(2,0,1)
        var=var.permute(2,0,1)
        D=(D.unsqueeze(0)).unsqueeze(0)
        Dt = D.permute(0, 1, 3, 2)
        Dty=torch.matmul(Dt,y).permute(3,1,2,0)
        DS_p=D*var.unsqueeze(2)
        # I = torch.eye(self.T,)
        S=torch.inverse(torch.matmul(DS_p,Dt)+(1/beta)*self.I)
        R=torch.matmul(torch.matmul(DS_p.permute(0,1,3,2),S), DS_p)
        Sigma=torch.diag_embed(var)-R
        mu=beta*torch.matmul(Sigma,Dty).squeeze(3)
        if self.mode =='sbl':
            alpha_loss=((mu**2-torch.diagonal(input=R, dim1=-2, dim2=-1))**2).mean()
        elif self.mode =='prior_learning':
            alpha_loss = torch.abs(mu).mean()
        else:
            print('Error !! Unknown mode')
        return mu, Sigma, alpha_loss

    def forward(self, x):
        dic = creatRealDictionary(self.T, self.rr, self.theta, self.gid)
        var =self.alpha_net(x)
        mu, Sigma, alpha_loss = self.compute_posterior(dic,var, x)
        return mu, Sigma, alpha_loss


class Decoder(nn.Module):
    def __init__(self, rr, theta, T, PRE, gpu_id):
        super(Decoder, self).__init__()

        self.rr = rr
        self.theta = theta
        self.T = T
        self.PRE = PRE
        self.gid = gpu_id

    def forward(self, x):
        dic = creatRealDictionary(self.T + self.PRE, self.rr, self.theta, self.gid)
        result = torch.matmul(dic, x)
        return result


class SBDyan(nn.Module):
    def __init__(self, Drr, Dtheta, T, PRE, gpu_id, args):
        super(SBDyan, self).__init__()
        self.beta = nn.Parameter(args.beta)
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.encoder = Encoder(self.rr, self.theta, T, gpu_id, self.beta, args)
        self.decoder = Decoder(self.rr, self.theta, T, PRE, gpu_id)

    def forward(self, x):
        mu, Sigma, alpha_loss=self.encoder(x)
        mu=mu.permute(1,2,0)
        y_hat=self.decoder(mu)
        return y_hat, mu, Sigma, alpha_loss

    # def forward2(self, x):
    #     return self.l1(x)

