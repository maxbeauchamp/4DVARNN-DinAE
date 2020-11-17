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
    with open('_PATH_/config.yml', 'rb') as f:
        conf = yaml.load(f.read())  

    # list of global parameters (comments to add)
    fileMod             	= datapath+'DATA/'+domain+conf['path_files']['fileMod']
    fileOI              	= [ datapath+'DATA/'+domain+x for x in conf['path_files']['fileOI'] ]
    fileObs             	= [ datapath+'DATA/'+domain+x for x in conf['path_files']['fileObs'] ]
    if opt=="nadir":
        fileObs         	= fileObs[0]
        fileOI          	= fileOI[0]
    elif opt=="swot": 
        fileObs         	= fileObs[1]
        fileOI          	= fileOI[1]
    else:
        fileObs         	= fileObs[2]
        fileOI          	= fileOI[2]
    flagTrWMissingData  	= conf['data_options']['flagTrWMissingData'] 
    flagloadOIData 		= conf['data_options']['flagloadOIData']
    include_covariates  	= conf['data_options']['include_covariates']
    lfile_cov                   = [ datapath+'DATA/'+domain+x for x in conf['data_options']['lfile_cov'] ]
    lname_cov                   = conf['data_options']['lname_cov']
    lid_cov                     = conf['data_options']['lid_cov'] 
    N_cov               	= ifelse(include_covariates==True,len(lid_cov),0)
    size_tw             	= conf['data_options']['size_tw'] 
    Wsquare     		= conf['data_options']['Wsquare']
    Nsquare     		= conf['data_options']['Nsquare']
    start_eval_index            = conf['data_options']['start_eval_index']
    end_eval_index              = conf['data_options']['end_eval_index']
    start_train_index           = conf['data_options']['start_train_index']
    end_train_index             = conf['data_options']['end_train_index']
    DimAE       		= conf['NN_options']['DimAE']
    flagAEType  		= conf['NN_options']['flagAEType']
    flag_MultiScaleAEModel      = conf['NN_options']['flag_MultiScaleAEModel']
    alpha                       = conf['loss_weighting']['alpha']
    alpha4DVar                  = conf['loss_weighting']['alpha4DVar']
    flagGradModel               = conf['solver_options']['flagGradModel']
    flagOptimMethod             = conf['solver_options']['flagOptimMethod']
    sigNoise        		= conf['data_options']['sigNoise']
    flagTrOuputWOMissingData    = conf['data_options']['flagTrOuputWOMissingData']
    stdMask              	= conf['data_options']['stdMask']
    flagDataWindowing 		= conf['data_options']['flagDataWindowing']
    dropout           		= conf['data_options']['dropout']
    wl2               		= conf['data_options']['wl2']
    flagLoadModel      		= conf['training_params']['flagLoadModel']
    batch_size        		= conf['training_params']['batch_size']
    if ( (torch.cuda.is_available()) and (torch.cuda.device_count()>1) ):
        batch_size = batch_size*torch.cuda.device_count()
    print("Batch size="+str(batch_size)+" on "+str(torch.cuda.device_count())+" GPUs")
    NbEpoc            		= conf['training_params']['NbEpoc']
    Niter			= conf['training_params']['Niter']

    # create the output directory
    if flagAEType==1:
        suf1="ConvAE",
    elif flagAEType==2:
        suf1="GENN"
    else:
        suf1="PINN"
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
    'stdMask','flagDataWindowing','dropout','wl2',\
    'start_eval_index','end_eval_index',\
    'start_train_index','end_train_index',\
    'batch_size','NbEpoc','Niter','flag_MultiScaleAEModel',\
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
