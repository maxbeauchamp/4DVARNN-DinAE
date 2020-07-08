#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .mods.utils.utils_solver.model_GradUpdate0    import model_GradUpdate0
from .mods.utils.utils_solver.model_GradUpdate1    import model_GradUpdate1
from .mods.utils.utils_solver.model_GradUpdate2    import model_GradUpdate2
from .mods.utils.utils_solver.model_GradUpdate3    import model_GradUpdate3

def replace_tup_at_idx(tup, idx, val):
    lst = list(tup)
    lst[idx] = val
    return tuple(lst)

# NN architecture based on given dynamical NN prior (model_AE)
# solving the reconstruction of the hidden states using a number of 
# fixed-point iterations prior to a gradient-based minimization
class Model_4DVarNN_GradFP(torch.nn.Module):
    def __init__(self,mod_AE,ShapeData,NiterProjection,NiterGrad,GradType,OptimType,InterpFlag=False,periodicBnd=False,N_cov=0):
        super(Model_4DVarNN_GradFP, self).__init__()
        self.model_AE = mod_AE
        with torch.no_grad():
            print('Optim type %d'%OptimType)
            self.OptimType   = OptimType
            self.NProjFP     = int(NiterProjection)
            self.NGrad       = int(NiterGrad)
            self.Ncov        = N_cov
            self.InterpFlag  = InterpFlag
            self.periodicBnd = periodicBnd
            self.shape       = ShapeData
        # Define Solver type according to OptimType
        ## Gradient-based minimization using a fixed-step descent
        if OptimType == 0:
            self.model_Grad = model_GradUpdate0(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType)
        ## Gradient-based minimization using a CNN using a (sub)gradient as inputs 
        elif OptimType == 1:
            self.model_Grad = model_GradUpdate1(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType,self.periodicBnd)
        ## Gradient-based minimization using a LSTM using a (sub)gradient as inputs
        elif OptimType == 2:
            self.model_Grad = model_GradUpdate2(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType,self.periodicBnd)
        elif OptimType == 3:
            self.model_Grad = model_GradUpdate2(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType,30)
                
    def forward(self, x_inp,xobs,mask,g1=None,g2=None,normgrad=0.0):
        mask_  = torch.add(1.0,torch.mul(mask,-1.0)) #1. - mask
        x      = torch.mul(x_inp,1.0)

        # new index to select appropriate data if covariates are used
        index = np.arange(0,self.shape[0],self.Ncov+1)  

        # fixed-point iterations
        if self.NProjFP > 0:
            for kk in range(0,self.NProjFP):        
                x_proj   = self.model_AE(x)
                x_proj   = torch.mul(x_proj,mask_[index])
                x[index] = torch.mul(x[index], mask[index])   
                x[index] = torch.add(x[index], x_proj)

        # gradient iteration
        if self.NGrad > 0:
            # gradient normalisation
            grad     = self.model_Grad.compute_Grad(x[index], self.model_AE(x),xobs,mask[index])
            if normgrad == 0. :
                _normgrad = torch.sqrt( torch.mean( grad**2 ) )
            else:
                _normgrad = normgrad
            for kk in range(0,self.NGrad):
                # AE prediction
                xpred = self.model_AE(x)
                # gradient update
                ## Gradient-based minimization using a fixed-step descent
                if self.OptimType == 0:
                    grad  = self.model_Grad( x[index], xpred, xobs, mask[index] , _normgrad )
                ## Gradient-based minimization using a CNN using a (sub)gradient as inputs
                elif self.OptimType == 1:               
                    if kk == 0:
                        grad  = self.model_Grad( x[index], xpred, xobs, mask[index], g1 , _normgrad)
                    else:
                        grad  = self.model_Grad( x[index], xpred, xobs, mask[index], grad_old , _normgrad)
                  grad_old = torch.mul(1.,grad)
                ## Gradient-based minimization using a LSTM using a (sub)gradient as inputs
                elif self.OptimType == 2:               
                    if kk == 0:
                      grad,hidden,cell  = self.model_Grad( x[index], xpred, xobs, mask[index], g1, g2 , _normgrad )
                    else:
                      grad,hidden,cell  = self.model_Grad( x[index], xpred, xobs, mask[index], hidden, cell , _normgrad )
                # interpolation constraint
                if( self.InterpFlag == True ):
                    # optimization update
                    xnew = x[index] - grad
                    x[index]    = x[index] * mask[index] + xnew * mask_[index]
                else:
                    # optimization update
                    x[index] = x[index] - grad

            if self.OptimType == 1:
                return x,grad_old,_normgrad
            if self.OptimType == 2:
                return x,hidden,cell,_normgrad
            else:
                return x,_normgrad
        else:
            _normgrad = 1.
            if self.OptimType == 1:
                return x,None,_normgrad
            if self.OptimType == 2:
                return x,None,None,_normgrad
            else:
                return x,_normgrad
