#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gradient computation: subgradient assuming a prior of the form ||x-f(x)||^2 vs. autograd

class Compute_Grad(torch.nn.Module):
    def __init__(self,ShapeData,GradType):
        super(Compute_Grad, self).__init__()
        with torch.no_grad():
            self.GradType  = GradType
            self.shape     = ShapeData        
        self.alphaObs    = torch.nn.Parameter(torch.Tensor([1.]))
        self.alphaAE     = torch.nn.Parameter(torch.Tensor([1.]))
        if( self.GradType == 3 ):
            self.NbEngTerms    = 3            
            self.alphaEngTerms = torch.nn.Parameter(torch.Tensor(np.ones((self.NbEngTerms,1))))
            self.alphaEngTerms.requires_grad = True
        if( self.GradType == 2 ):
            self.alphaL1    = torch.nn.Parameter(torch.Tensor([1.]))
            self.alphaL2    = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self,x,xpred,xobs,mask):

        # compute gradient
        ## subgradient for prior ||x-g(x)||^2 
        if self.GradType == 0: 
          grad  = torch.add(xpred,-1.,x)
          grad2 = torch.add(x,-1.,xobs)
          grad  = torch.add(grad,1.,grad2)
          grad  = self.alphaAE * grad + self.alphaObs * grad2
        ## true gradient using autograd for prior ||x-g(x)||^2 
        elif self.GradType == 1: 
          loss1 = torch.mean( (xpred - x)**2 )
          loss2 = torch.sum( (xobs - x)**2 * mask ) / torch.sum( mask )
          loss  = self.alphaAE**2 * loss1 + self.alphaObs**2 * loss2
          grad = torch.autograd.grad(loss,x,create_graph=True)[0]
        ## true gradient using autograd for prior ||x-g(x)||
        elif self.GradType == 2: 
          loss1 = self.alphaL2**2 * torch.mean( (xpred - x)**2 ) +\
                  self.alphaL1**2 * torch.mean( torch.abs(xpred - x) )
          loss2 = torch.sum( (xobs - x)**2 * mask ) / torch.sum( mask )
          loss  = self.alphaAE**2 * loss1 + self.alphaObs**2 * loss2
          grad = torch.autograd.grad(loss,x,create_graph=True)[0]
        ## true gradient using autograd for prior ||x-g1(x)||^2 + ||x-g2(x)||^2 
        elif self.GradType == 3: 
          if len(self.shape) == 2 :            
            for ii in range(0,xpred.size(1)):
               if( ii == 0 ):
                   loss1 = self.alphaEngTerms[ii]**2 * torch.mean( (xpred[:,ii,:,:].view(-1,self.shape[0],self.shape[1]) - x)**2 )
               else:
                   loss1 += self.alphaEngTerms[ii]**2 * torch.mean( (xpred[:,ii,:,:].view(-1,self.shape[0],self.shape[1]) - x)**2 )
          else:
               if( ii == 0 ):
                   loss1 = self.alphaEngTerms[ii]**2 * torch.mean( (xpred[:,0:self.shape[0],:,:] - x)**2 )
               else:
                   loss1 += self.alphaEngTerms[ii]**2 * torch.mean( (xpred[:,ii*self.shape[0]:(ii+1)*self.shape[0],:,:] - x)**2 )
              
          loss2 = torch.sum( (xobs - x)**2 * mask ) / torch.sum( mask )
          loss  = self.alphaAE**2 * loss1 + self.alphaObs**2 * loss2
          grad = torch.autograd.grad(loss,x,create_graph=True)[0]
        ## true gradient using autograd for prior ||g(x)||^2 
        elif self.GradType == 4: 
          loss1 = torch.mean( xpred **2 )
          loss2 = torch.sum( (xobs - x)**2 * mask ) / torch.sum( mask )
          loss  = self.alphaAE**2 * loss1 + self.alphaObs**2 * loss2
          grad = torch.autograd.grad(loss,x,create_graph=True)[0]

        # Check is this is needed or not
        grad.retain_grad()

        return grad
