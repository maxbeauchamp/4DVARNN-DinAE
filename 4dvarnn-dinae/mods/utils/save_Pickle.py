import os
from .tools import *
from .graphics import *

def save_Pickle(dirSAVE,\
                x_train,x_train_missing,x_train_pred,rec_AE_Tr,meanTr,stdTr,\     
                x_test,x_test_missing,x_test_pred,rec_AE_Tt):     

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    saved_path = dirSAVE+'/saved_path_%03d'%(iter)+'_'+suf1+'_'+suf2+'.pickle'

    ## keep only the information on the target variable (remove covariates)
    if include_covariates == True:
        index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
        x_train         = np.moveaxis(x_train[:,index,:,:]
        x_train_missing = np.moveaxis(x_train_missing[:,index,:,:]
        x_test         = x_test[:,index,:,:]
        x_test_missing = x_test_missing[:,index,:,:]
        meanTr = meanTr[0]
        stdTr  = stdTr[0]

    ## reshape and rescale variables
    x_train         = meanTr+ np.moveaxis(x_train,1,3)*stdTr
    x_train_missing = meanTr+ np.moveaxis(x_train_missing,1,3)*stdTr
    x_train_pred    = meanTr+ np.moveaxis(x_train_pred,1,3)*stdTr
    rec_AE_Tr       = meanTr+ np.moveaxis(rec_AE_Tr,1,3)*stdTr
    x_test          = meanTr+ np.moveaxis(x_test,1,3)*stdTr
    x_test_missing  = meanTr+ np.moveaxis(x_test_missing,1,3)*stdTr
    x_test_pred     = meanTr+ np.moveaxis(x_test_pred,1,3)*stdTr
    rec_AE_Tt       = meanTr+ np.moveaxis(rec_AE_Tt,1,3)*stdTr

    ## add OI (large-scale) to state if required
    if flagloadOIData == 1::
        x_train         = x_train + x_train_OI
        x_train_missing = x_train_missing + x_train_OI
        x_train_pred    = x_train_pred + x_train_OI
        rec_AE_Tr       = rec_AE_Tr + x_train_OI
        x_test          = x_test + x_test_OI
        x_test_missing  = x_test_missing + x_test_OI
        x_test_pred     = x_test_pred + x_test_OI
        rec_AE_Tt       = rec_AE_Tt + x_test_OI

    idT = int(np.floor(x_test.shape[3]/2))
    with open(saved_path, 'wb') as handle:
        pickle.dump([x_train[:,:,:,idT],\
                     x_train_missing[:,:,:,idT],\
                     x_train_pred[:,:,:,idT],\
                     rec_AE_Tr[:,:,:,idT],\
                     x_test[:,:,:,idT],\
                     x_test_missing[:,:,:,idT],\
                     x_test_pred[:,:,:,idT],\
                     rec_AE_Tt[:,:,:,idT],\
                    ], handle)
