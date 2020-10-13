#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

from skfda.representation.basis import BSpline as BSpline
import numpy as np
import torch
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from BsCNN import BsCNN

## Compute spatial_advection_diffusion_kernel
def AD_filter_3x3(alpha,
                  bspl_basis_alongX,
                  bspl_basis_alongY,
                  bspl_basis_alongT,
                  s,t):
# alpha is the tensor of size #(Number of Bsplines coefficients) x #(Number of parameters=6)
# s is a 2D-vector giving location in space
# t is a scalar giving location along the time-window

    # compute 3D B-splines basis functions
    bX=bspl_basis_alongX(s[0])[:,:,-1]
    bY=bspl_basis_alongY(s[1])[:,:,-1]
    bT=bspl_basis_alongT(t)[:,:,-1]
    bXY=np.tensordot(bX,bY,axes=0).swapaxes(2,1).reshape(bX.shape[0]*bY.shape[0],
                                                         bX.shape[1]*bY.shape[1])
    bXYT=np.tensordot(bXY,bT,axes=0).swapaxes(2,1).reshape(bXY.shape[0]*bT.shape[0],
                                                         bXY.shape[1]*bT.shape[1])
    bXYT=torch.from_numpy(bXYT).to(device)

    # Compute Kappa value
    kappa = torch.sum(bXYT[:,-1]*alpha[0])
    # Compute H diffusion tensor
    gamma = torch.sum(bXYT[:,-1]*alpha[1])
    vx    = torch.sum(bXYT[:,-1]*alpha[2])
    print(vx)
    print(torch.cat([vx,vy]))
    print(torch.transpose(torch.cat([vx,vy]),0,1))
    print('toto2')
    vy    = torch.sum(bXYT[:,-1]*alpha[3])
    H     = gamma*torch.eye(2)+torch.mul(torch.cat([vx,vy]),torch.transpose(torch.cat([vx,vy]),0,1))
    # Compute m advection vector
    m1    = torch.sum(bXYT[:,-1]*alpha[4])
    m2    = torch.sum(bXYT[:,-1]*alpha[5])
    m     = torch.cat([m1,m2])

    # Define w_G stencil
    weights = torch.empty(3,3)       
    weights[0,0] = H[0,1]/2
    weights[0,1] = -1.*H[1,1] - m[1]
    weights[0,2] = -1.*H[0,1]/2
    weights[1,0] = -1.*H[0,0] - m[0]
    weights[1,1] = kappa**2 + 2.*(H[0,0]+H[1,1])
    weights[1,2] = -1.*H[0,0] + m[0]
    weights[2,0] = -1.*H[0,1]/2
    weights[2,1] = -1.*H[1,1] + m[1]
    weights[2,2] = H[0,1]/2
    return weights

## Convolve two kernels of different sizes
def convolve_kernels(k1, k2):
# input k1: A tensor of shape ``(out1, in1, s1, s1)``
# input k2: A tensor of shape ``(out2, in2, s2, s2)``
# returns: A tensor of shape ``(out2, in1, s1+s2-1, s1+s2-1)``
#          so that convolving with it equals convolving with k1 and
#          then with k2.
    padding = k2.shape[-1] - 1
    # Flip because this is actually correlation, and permute to adapt to BHCW
    k3 = torch.conv2d(k1.permute(1, 0, 2, 3), k2.flip(-1, -2),
                      padding=padding).permute(1, 0, 2, 3)
    return k3

# Physics informed Conv2D Layer deriving from spatial advection-diffusion equation
def Space_Time_ADConv2d(dict_global_Params,genFilename,shapeData,shapeBsBasis):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    globals()['shapeData']=shapeData
    globals()['shapeBsBasis']=shapeBsBasis
    
    class Encoder(torch.nn.Module):

        def __init__(self,):
        # model_Alpha is the NN to estimate B-splines basis coefficients
        # Nx, Ny, Nt are the dimensions along X, Y and T
        # Nfbx, Nfby, Nfbt are the number of B-splines basis along X, Y and T
            super(Encoder, self).__init__()
            self.model_Alpha = BsCNN(DimAE,shapeData,shapeBsBasis)
            self.Nx      = shapeData[2]
            self.Ny      = shapeData[1]
            self.Nt      = shapeData[0]
            self.Nfbx    = shapeBsBasis[2]
            self.Nfby    = shapeBsBasis[1]
            self.Nfbt    = shapeBsBasis[0]
            # 3D B-splines basis functions
            self.bspl_basis_alongX = BSpline(domain_range=[-1,self.Nx+1], n_basis=self.Nfbx, order=3)
            self.bspl_basis_alongY = BSpline(domain_range=[-1,self.Ny+1], n_basis=self.Nfby, order=3)
            # time domain of definition: [t-Ntdt,t]
            self.bspl_basis_alongT = BSpline(domain_range=[-1*self.Nt,1], n_basis=self.Nfbt, order=3)
            # define 3*3 identity kernel
            self.ID_kernel=torch.tensor([[0., 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 0.]])

        def kernel_ij(self,alpha,i,j,s):
            # 3*3 init kernel
            t = (-1.*(self.Nt-1)) + i
            kernel = self.ID_kernel + AD_filter_3x3(alpha,
                                           self.bspl_basis_alongX,
                                           self.bspl_basis_alongY,
                                           self.bspl_basis_alongT,
                                           s,t)
            # loop
            for k in np.arange(i+1,j):
                t = (-1.*(self.Nt-1)) + k
                kernel = convolve_kernels(kernel,
                                      self.ID_kernel + AD_filter_3x3(alpha,
                                                    self.bspl_basis_alongX,
                                                    self.bspl_basis_alongY,
                                                    self.bspl_basis_alongT,
                                                    s,t))
            return kernel

        def forward(self, x):
        # Necessary forward computations
        # x has shape (#time,#x,#y)
            target = torch.empty(x.shape)
            # compute the coefficients alpha of size (N_coeff*6)
            alpha = self.model_Alpha(x)
            # loop over space
            for Ix,Iy in list(itertools.product(range(x.shape[2]),range(x.shape[1]))):
            # loop over time
                for j in range(x.shape[0]):
                    for i in range(x.shape[0]):
                        # compute kernel weights
                        cpt=0. 
                        if ( (j>i) and (i!=j) ):
                            weights =  self.kernel_ij(alpha,i,j,s=[Ix,Iy])
                            # define inputs (see edge effect)
                            inputs = torch.zeros(filter.shape)
                            ix_min = max(0,Ix-filter.shape[0])
                            ix_max = min(self.Nx,Ix+filter.shape[0]+1)
                            iy_min = max(0,Iy-filter.shape[1])
                            iy_max = min(self.Ny,Iy+filter.shape[1]+1)
                            inputs = x[i,ix_min:ix_max,iy_min:iy_max]
                            target[j,Ix,Iy]+=torch.nn.functional.conv2d(inputs,weights,bias=None,stride=2,padding=0)
                        cpt+=1
                    target[j,Ix,Iy]/=cpt		  
            return target

    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()

        def forward(self, x):
            return torch.mul(1.,x)

    return Encoder, Decoder

