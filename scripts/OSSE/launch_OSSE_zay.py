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

# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    with open('_PATH_/config_zay.yml', 'rb') as f:
        conf = yaml.load(f.read())

    opt        = conf['data_options']['opt']
    lag        = str(conf['data_options']['lag'])
    domain     = conf['data_options']['domain']
    type_obs   = conf['data_options']['type_obs']

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
    lfile_cov                   = [ datapath+'DATA/domain='+domain+x for x in conf['data_options']['lfile_cov'] ]
    lname_cov                   = conf['data_options']['lname_cov']
    lid_cov                     = conf['data_options']['lid_cov'] 
    N_cov               	= ifelse(include_covariates==True,len(lid_cov),0)
    size_tw             	= conf['data_options']['size_tw']
    dwscale                     = conf['data_options']['dwscale']
    start_eval_index            = conf['data_options']['start_eval_index']
    end_eval_index              = conf['data_options']['end_eval_index']
    start_train_index           = conf['data_options']['start_train_index']
    end_train_index             = conf['data_options']['end_train_index']
    DimAE       		= conf['NN_options']['DimAE']
    flagAEType  		= conf['NN_options']['flagAEType']
    alpha                       = conf['loss_weighting']['alpha']
    alpha4DVar                  = conf['loss_weighting']['alpha4DVar']
    solver_type                 = conf['solver_options']['solver_type']
    flagGradModel               = conf['solver_options']['flagGradModel']
    flagOptimMethod             = conf['solver_options']['flagOptimMethod']
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
    suf3 = ifelse(solver_type=="GB","GB"+str(flagOptimMethod),"FP")
    suf4 = ifelse(include_covariates==True,"w"+'-'.join(lid_cov),"wocov")
    dirSAVE = ifelse(opt!='swot',\
              scratchpath+domain+'/OSSE/resIA_'+opt+'_nadlag_'+lag+"_"+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'_'+suf4+'/',\
              scratchpath+domain+'/OSSE/resIA_'+opt+'_'+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'_'+suf4+'/')
    if not os.path.exists(dirSAVE):
        mk_dir_recursive(dirSAVE)
    #else:
    #    shutil.rmtree(dirSAVE)
    #    mk_dir_recursive(dirSAVE)

    # push all global parameters in a list
    def createGlobParams(params):
        return dict(((k, eval(k)) for k in params))
    list_globParams=['domain','fileMod','fileObs','fileOI',\
    'include_covariates','N_cov','lfile_cov','lid_cov','lname_cov',\
    'flagTrWMissingData','flagloadOIData',\
    'size_tw','dwscale',\
    'DimAE','flagAEType','flagLoadModel',\
    'solver_type','flagOptimMethod','flagGradModel',\
    'alpha','alpha4DVar',\
    'start_eval_index','end_eval_index',\
    'start_train_index','end_train_index',\
    'batch_size','NbEpoc','Niter',\
    'dirSAVE','suf1','suf2','suf3','suf4']
    globParams = createGlobParams(list_globParams)   

    #1) *** Read the data ***
    genFilename, \
    input_train,mask_train,target_train,x_train_OI,lday_pred,meanTr, stdTr,\
    input_test,mask_test,target_test,x_test_OI,lday_test = import_Data_OSSE(globParams,type_obs)

    #2) *** Define AE architecture ***
    shapeData=(input_train.shape[3],input_train.shape[1],input_train.shape[2])
    genFilename, encoder, decoder, model_AE, DIMCAE = define_Models(globParams,genFilename,shapeData)

    #5) *** Train ConvAE ***      
    learning_OSSE(globParams,genFilename,\
                  input_train,mask_train,target_train,x_train_OI,lday_pred,meanTr,stdTr,\
                  input_test,mask_test,target_test,x_test_OI,lday_test,model_AE,DIMCAE)
