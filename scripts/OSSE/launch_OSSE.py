#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:45 2019

@author: rfablet, mbeaucha
"""

from dinae_4dvarnn import *

def ifelse(cond1,val1,val2):
    if cond1==True:
        res = val1
    else:
        res = val2
    return res

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# description of the parameters
opt      = sys.argv[1]            # nadir/swot/nadirswot
lag      = sys.argv[2]            # 0...5
type_obs = sys.argv[3]            # mod/obs
domain   = sys.argv[4]            # OSMOSIS/GULFSTREAM
wMis     = int(sys.argv[5])       # 0/1/2
wCov     = str2bool(sys.argv[6])  # False/True

# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # list of global parameters (comments to add)
    fileMod                     = datapath+domain+"/ref/NATL60-CJM165_"+domain+"_ssh_y2013.1y.nc" # Model file
    fileOI                      = datapath+domain+"/oi/ssh_NATL60_4nadir.nc"            # OI file
    if opt=="nadir":
        fileObs                 = datapath+domain+"/data/gridded_data_swot_wocorr/dataset_nadir_"+lag+"d.nc" # Obs file (1)
    elif opt=="swot": 
        fileObs                 = datapath+domain+"/data/gridded_data_swot_wocorr/dataset_swot.nc"           # Obs file (2)
    else:
        fileObs                 = datapath+domain+"/data/gridded_data_swot_wocorr/dataset_nadir_"+lag+"d_swot.nc" # Obs file (3)
    flagTrWMissingData          = wMis  # Training phase with or without missing data
    flagloadOIData 		= 1     # load OI: work on rough variable or anomaly
    include_covariates          = wCov  # use additional covariates in initial layer
    if include_covariates==True:
        '''lfile_cov            = [datapath+domain+"/ref/NATL60-CJM165_sst_y2013.1y.nc",\
                                       datapath+domain+"/ref/NATL60-CJM165_sss_y2013.1y.nc",\
                                       datapath+domain+"/oi/ssh_NATL60_4nadir.nc"]
        lname_cov               = ["sst","sss","ssh_mod"]
        lid_cov                 = ["SST","SSS","OI"]'''
        lfile_cov               = [datapath+domain+"/oi/ssh_NATL60_4nadir.nc"]
        lname_cov               = ["ssh_mod"]
        lid_cov                 = ["OI"]
        N_cov                   = len(lid_cov)
    else:
        lfile_cov               = [""]
        lname_cov               = [""]
        lid_cov                 = [""]
        N_cov                   = 0
    size_tw                     = 11    # Length of the 4th dimension          
    Wsquare     		= 4     # half-width of holes
    Nsquare     		= 3     # number of holes
    DimAE       		= 40    # Dimension of the latent space
    flagAEType  		= 2     # model type, ConvAE or GE-NN
    flagLoadModel               = 0     # load pre-defined AE model or not
    flag_MultiScaleAEModel      = 0     # see flagProcess2_7: work on HR(0), LR(1), or HR+LR(2)
    # alpha ?? weights for loss function
    alpha                       = np.array([1.,0.1])
    alpha4DVar                  = np.array([0.01,1.])
    flagGradModel               = 2     # Gradient computation (0: subgradient, 1: true gradient/autograd)
    flagOptimMethod             = 2     # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
    sigNoise        		= 1e-1
    flagUseMaskinEncoder 	= 0
    flagTrOuputWOMissingData    = 1
    stdMask              	= 0.
    flagDataWindowing 		= 2  # 2 for SSH case-study
    dropout           		= 0.0
    wl2               		= 0.0000
    batch_size        		= 4
    NbEpoc            		= 20
    Niter = ifelse(flagTrWMissingData==1,20,20)

    # create the output directory
    suf1 = ifelse(flagAEType==1,"ConvAE","GENN")
    if flagTrWMissingData==0:
        suf2 = "womissing"
    elif flagTrWMissingData==1:
        suf2 = "wmissing"
    else:
        suf2 = "wwmissing"
    suf3 = "GB"+str(flagOptimMethod)
    suf4 = ifelse(include_covariates==True,"w"+'-'.join(lid_cov),"wocov")
    dirSAVE = ifelse(opt!='swot',\
              '/gpfsscratch/rech/yrf/uba22to/4DVARNN-DINAE/'+domain+'/resIA_'+opt+'_nadlag_'+lag+"_"+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'_'+suf4+'/',\
              '/gpfsscratch/rech/yrf/uba22to/4DVARNN-DINAE/'+domain+'/resIA_'+opt+'_'+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'_'+suf4+'/')
    if not os.path.exists(dirSAVE):
        mk_dir_recursive(dirSAVE)
    else:
        shutil.rmtree(dirSAVE)
        mk_dir_recursive(dirSAVE)

    # push all global parameters in a list
    def createGlobParams(params):
        return dict(((k, eval(k)) for k in params))
    list_globParams=['domain','fileMod','fileObs','fileOI',\
    'include_covariates','N_cov','lfile_cov','lid_cov','lname_cov',\
    'flagTrOuputWOMissingData','flagTrWMissingData',\
    'flagloadOIData','size_tw','Wsquare',\
    'Nsquare','DimAE','flagAEType','flagLoadModel',\
    'flagOptimMethod','flagGradModel','alpha','alpha4DVar','sigNoise',\
    'flagUseMaskinEncoder','stdMask',\
    'flagDataWindowing','dropout','wl2','batch_size',\
    'NbEpoc','Niter','flag_MultiScaleAEModel',\
    'dirSAVE','suf1','suf2','suf3','suf4']
    globParams = createGlobParams(list_globParams)   

    #1) *** Read the data ***
    genFilename, \
    x_train,y_train,mask_train,gt_train,x_train_missing,x_train_OI,lday_pred,meanTr, stdTr,\
    x_test,y_test,mask_test,gt_test,x_test_missing,x_test_OI,lday_test = import_Data_OSSE(globParams,type_obs)

    #2) *** Define AE architecture ***
    shapeData=(x_train.shape[3],x_train.shape[1],x_train.shape[2])
    genFilename, encoder, decoder, model_AE, DIMCAE = define_Models(globParams,genFilename,shapeData)

    #5) *** Train ConvAE ***      
    learning_OSSE(globParams,genFilename,\
                  x_train,x_train_missing,mask_train,gt_train,x_train_OI,lday_pred,meanTr,stdTr,\
                  x_test,x_test_missing,mask_test,gt_test,x_test_OI,lday_test,model_AE,DIMCAE)
