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

# NN architecture based on given dynamical NN prior (model_AE)
# solving the reconstruction of the hidden states using a number of 
# fixed-point iterations prior to a gradient-based minimization
class Model_4DVarNN_Grad(torch.nn.Module):
    def __init__(self,mod_AE,ShapeData,NiterGrad,GradType,OptimType,InterpFlag=False,periodicBnd=False):
    #def __init__(self,mod_AE,GradType,OptimType):
        super(Model_4DVarNN_Grad, self).__init__()
        self.model_AE = mod_AE
        self.OptimType = OptimType
        self.GradType  = GradType
        self.InterpFlag = InterpFlag
        self.periodicBnd = periodicBnd
        
        if OptimType == 0:
          self.model_Grad = model_GradUpdate0(ShapeData,GradType)
        elif OptimType == 1:
          self.model_Grad = model_GradUpdate1(ShapeData,GradType,self.periodicBnd)
        elif OptimType == 2:
          self.model_Grad = model_GradUpdate2(ShapeData,GradType,self.periodicBnd)
        elif OptimType == 3:
          self.model_Grad = model_GradUpdate3(ShapeData,GradType,30)
        elif OptimType == 4:
          self.model_Grad = model_GradUpdate4(ShapeData,GradType,self.periodicBnd)
                    
        with torch.no_grad():
            #print('Opitm type %d'%OptimType)
            self.OptimType = OptimType
            #self.NProjFP   = int(NiterProjection)
            self.NGrad     = int(NiterGrad)
            
        
    def forward(self, x_inp,xobs,mask,g1=None,g2=None,normgrad=0.):
        if( self.InterpFlag == True ):
            #mask   = torch.add(1.0,torch.mul(mask,0.0)) # set mask to 1 # debug
            mask_  = torch.add(1.0,torch.mul(mask,-1.0)) #1. - mask
        x      = torch.mul(x_inp,1.0)

        # gradient iteration
        for kk in range(0,self.NGrad):
            # gradient normalisation
            grad     = self.model_Grad.compute_Grad(x, self.model_AE(x),xobs,mask)
            if normgrad == 0. :
                _normgrad = torch.sqrt( torch.mean( grad**2 ) )
            else:
                _normgrad = normgrad

            # AE pediction
            xpred = self.model_AE(x)
         
            # gradient update
            if self.OptimType == 0:
                grad  = self.model_Grad( x, xpred, xobs, mask, _normgrad )
    
            elif self.OptimType == 1:               
              if kk == 0:
                grad  = self.model_Grad( x, xpred, xobs, mask, g1, _normgrad )
              else:
                grad  = self.model_Grad( x, xpred, xobs, mask,grad_old, _normgrad )
              grad_old = torch.mul(1.,grad)
    
            elif self.OptimType == 2:               
              if kk == 0:
                grad,hidden,cell  = self.model_Grad( x, xpred, xobs, mask, g1, g2, _normgrad )
              else:
                grad,hidden,cell  = self.model_Grad( x, xpred, xobs, mask, hidden, cell, _normgrad )
    
            # interpolation constraint
            if( self.InterpFlag == True ):
                # optimization update
                xnew = x - grad
                x    = x * mask + xnew * mask_
            else:
                # optimization update
                x = x - grad

        if self.OptimType == 1:
            return x,grad_old,_normgrad
        if self.OptimType == 2:
            return x,hidden,cell,_normgrad
        else:
            return x,_normgrad
