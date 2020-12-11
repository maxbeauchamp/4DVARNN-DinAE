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

    lag        = str(conf['data_options']['lag'])
    domain     = conf['data_options']['domain']
    load_Model = conf['NN_options']['load_model']

    # list of global parameters (comments to add)
    fileOI              	= datapath+'DATA/OSE/'+domain+conf['path_files']['fileOI']
    fileObs             	= datapath+'DATA/OSE/'+domain+conf['path_files']['fileObs']
    fileMod                     = datapath+'DATA/'+domain+conf['path_files']['fileMod']
    fileOI_Mod                  = datapath+'DATA/'+domain+conf['path_files']['fileOI_Mod']
    fileBFN                     = datapath+'DATA/OSE/'+domain+conf['path_files']['fileBFN']
    flagTrWMissingData  	= 2
    flagloadOIData 		= conf['data_options']['flagloadOIData']
    include_covariates  	= conf['data_options']['include_covariates']
    lfile_cov                   = [ datapath+'DATA/OSE/'+domain+x for x in conf['data_options']['lfile_cov'] ]
    lname_cov                   = conf['data_options']['lname_cov']
    lid_cov                     = conf['data_options']['lid_cov']
    lfile_cov_mod               = [ datapath+'DATA/'+domain+x for x in conf['data_options']['lfile_cov_mod'] ]
    lname_cov_mod               = conf['data_options']['lname_cov_mod']
    lid_cov_mod                 = conf['data_options']['lid_cov_mod']
    N_cov               	= ifelse(include_covariates==True,len(lid_cov),0)
    size_tw             	= conf['data_options']['size_tw'] 
    dwscale                     = conf['data_options']['dwscale']
    DimAE       		= conf['NN_options']['DimAE']
    flagAEType  		= conf['NN_options']['flagAEType']
    alpha                       = conf['loss_weighting']['alpha']
    alpha4DVar                  = conf['loss_weighting']['alpha4DVar']
    solver_type                 = conf['solver_options']['solver_type']
    flagGradModel               = conf['solver_options']['flagGradModel']
    flagOptimMethod             = conf['solver_options']['flagOptimMethod']
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
    suf5 = ifelse(load_Model==True,"wotrain","wtrain")
    dirSAVE = scratchpath+domain+'/OSE/resIA_nadir_nadlag_'+lag+"_obs/"+suf3+'_'+suf1+'_'+suf2+'_'+suf4+'_'+suf5+'/'
    if not os.path.exists(dirSAVE):
        mk_dir_recursive(dirSAVE)
    #else:
    #    shutil.rmtree(dirSAVE)
    #    mk_dir_recursive(dirSAVE)

    # push all global parameters in a list
    def createGlobParams(params):
        return dict(((k, eval(k)) for k in params))
    list_globParams=['lag','domain','load_Model',
    'fileObs','fileOI','fileMod','fileOI_Mod','fileBFN',\
    'include_covariates','N_cov',
    'lfile_cov','lid_cov','lname_cov',\
    'lfile_cov_mod','lid_cov_mod','lname_cov_mod',\
    'flagTrWMissingData','flagloadOIData',\
    'size_tw','dwscale',\
    'DimAE','flagAEType',\
    'solver_type','flagOptimMethod','flagGradModel',\
    'alpha','alpha4DVar',\
    'batch_size','NbEpoc','Niter',\
    'dirSAVE','suf1','suf2','suf3','suf4']
    globParams = createGlobParams(list_globParams)   

    #1) *** Read the data ***
    genFilename, meanTr, stdTr,\
    x_inputs_train, mask_inputs_train,\
    x_targets_train, mask_targets_train,\
    lday_train, x_train_OI, x_mod, mask_mod, x_BFN = import_Data_OSE(globParams)

    #2) *** Define AE architecture ***
    shapeData=(x_inputs_train.shape[3],x_inputs_train.shape[1],x_inputs_train.shape[2])
    genFilename, encoder, decoder, model_AE, DIMCAE = define_Models(globParams,genFilename,shapeData)

    #3) *** Train ConvAE ***      
    learning_OSE(globParams,genFilename,meanTr,stdTr,\
                  x_inputs_train,mask_inputs_train,x_targets_train,mask_targets_train,
                  x_train_OI,x_mod,mask_mod,x_BFN,lday_train,model_AE,DIMCAE)
