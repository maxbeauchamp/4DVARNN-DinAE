#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:45 2019

@author: rfablet, mbeaucha
"""

from 4dvarnn-dinae import *

def ifelse(cond1,val1,val2):
    if cond1==True:
        res = val1
    else:
        res = val2
    return res

lag      = sys.argv[1]
domain   = sys.argv[2]
  
# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # list of global parameters (comments to add)
    fileOI                      = datapath+"/OSE/"+domain+"/training/oi/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc" # OI file
    fileObs                     = datapath+"/OSE/"+domain+"/training/data/dataset_nadir_"+lag+"d.nc"         # Obs file (1)
    flagTrWMissingData          = 2     # Training phase with or without missing data
    flagloadOIData 		= 1     # load OI: work on rough variable or anomaly
    include_covariates          = True  # use additional covariates in initial layer
    N_cov                       = 1     # SST, SSS and OI
    lfile_cov                   = [datapath+"/OSE/"+domain+"/training/oi/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc"]
    lname_cov                   = ["ssh"]
    lid_cov                     = ["OI"]
    size_tw                     = 11    # Length of the 4th dimension          
    Wsquare     		= 4     # half-width of holes
    Nsquare     		= 3     # number of holes
    DimAE       		= 200   # Dimension of the latent space
    flagAEType  		= 2     # model type, ConvAE or GE-NN
    flagLoadModel               = 0     # load pre-defined AE model or not
    flag_MultiScaleAEModel      = 0     # see flagProcess2_7: work on HR(0), LR(1), or HR+LR(2)
    alpha_Losss                 = np.array([1.,0.1]) # ??
    flagGradModel               = 0     # Gradient computation (0: subgradient, 1: true gradient/autograd)
    flagOptimMethod             = 1     # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
    load_Model                  = False # use a pre-trained model or not
    sigNoise        		= 1e-1
    flagUseMaskinEncoder 	= 0
    flagTrOuputWOMissingData    = 1
    stdMask              	= 0.
    flagDataWindowing 		= 2     # 2 for SSH case-study
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
    suf3 = "GB"+flagOptimMethod
    suf4 = ifelse(include_covariates==True,"w"+'-'.join(lid_cov),"wocov")
    suf5 = ifelse(load_Model==True,"wotrain","wtrain")
    dirSAVE = '/gpfsscratch/rech/yrf/uba22to/4DVARNN-DinAE/OSE/'+domain+'/resIA_nadir_nadlag_'+lag+"_obs/"+suf3+'_'+suf1+'_'+suf2+'_'+suf4+'_'+suf5+'/'
    if not os.path.exists(dirSAVE):
        mk_dir_recursive(dirSAVE)
    #else:
    #    shutil.rmtree(dirSAVE)
    #    mk_dir_recursive(dirSAVE)

    # push all global parameters in a list
    def createGlobParams(params):
        return dict(((k, eval(k)) for k in params))
    list_globParams=['domain','fileObs','fileOI',\
    'include_covariates','N_cov','lfile_cov','lid_cov','lname_cov',\
    'flagTrOuputWOMissingData','flagTrWMissingData',\
    'flagloadOIData','size_tw','Wsquare',\
    'Nsquare','DimAE','flagAEType','flagLoadModel',\
    'flagOptimMethod','flagGradModel','alpha_Losss','sigNoise',\
    'load_Model','flagUseMaskinEncoder','stdMask',\
    'flagDataWindowing','dropout','wl2','batch_size',\
    'NbEpoc','Niter','flag_MultiScaleAEModel',\
    'dirSAVE','suf1','suf2','suf3','suf4']
    globParams = createGlobParams(list_globParams)   

    #1) *** Read the data ***
    genFilename, meanTt, stdTt,\
    x_test, y_test, mask_test, gt_test, x_test_missing, lday_test, x_test_OI = import_Data_OSE(globParams)

    #2) *** Define AE architecture ***
    genFilename, encoder, decoder, model_AE, DIMCAE = define_Models(globParams,genFilename,x_test,mask_test)

    #5) *** Train ConvAE ***      
    learning_OSE(globParams,genFilename,meanTt,stdTt,\
                 x_test,x_test_missing,mask_test,gt_test,x_test_OI,lday_test,\
                 encoder,decoder,model_AE,DIMCAE)

