from dinae_4dvarnn import *

def Imputing_NaN(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell
    """
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def Imputing_NaN_3d(data, invalid=None):
    data_wonan = data
    for i in range(data.shape[0]):
        data_wonan[i,:,:] = Imputing_NaN(data[i,:,:])
    return data_wonan

def ndarray_NaN(shape):
    arr    = np.empty(shape)
    arr[:] = np.nan
    return arr

def import_Data_OSSE(dict_global_Params,type_obs):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    #*** Start reading the data ***#
    indN_Tt = np.concatenate([np.arange(60,80),np.arange(140,160),\
                             np.arange(220,240),np.arange(300,320)])
    indN_Tr = np.delete(range(365),indN_Tt)
    #indN_Tt = np.arange(start_eval_index,end_eval_index)   # index of evaluation period
    #indN_Tr = np.arange(start_train_index,end_train_index) # index of training period
    lday_pred=[ datetime.strftime(datetime.strptime("2012-10-01",'%Y-%m-%d')\
                          + timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tr ]
    lday_test=[ datetime.strftime(datetime.strptime("2012-10-01",'%Y-%m-%d')\
                          + timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tt ]

    if domain=="OSMOSIS":
        indLat     = np.arange(0,200)
        indLon     = np.arange(0,160)         
    else:
        indLat     = np.arange(0,200)
        indLon     = np.arange(0,200)     

    #*** TRAINING DATASET ***#
    print("1) .... Load SSH dataset (training data): "+fileObs)

    nc_data_mod = Dataset(fileMod,'r')
    nc_data_obs = Dataset(fileObs,'r')    
    # DAILY NATL60
    x_mod = Imputing_NaN_3d(np.copy(nc_data_mod['ssh'][:,indLat,indLon]))
    # NADIR/SWOT observations
    if flagTrWMissingData==0:
        x_obs= x_mod
    else:
        x_obs = np.copy(nc_data_obs['ssh_'+type_obs][:,indLat,indLon])
        mask = np.asarray(~np.isnan(x_obs))
    if flagTrWMissingData==0:
        mask[indN_Tr,:,:]  = 1
    nc_data_mod.close()
    nc_data_obs.close()
    # load OI data
    if flagloadOIData == 1:
        print(".... Load OI SSH dataset (training data): "+fileOI)
        nc_data    = Dataset(fileOI,'r')
        x_OI = Imputing_NaN_3d(np.copy(nc_data['ssh_'+type_obs][:,indLat,indLon]))
        nc_data.close()
    # load covariates
    if include_covariates==True:
        cov=[]
        for icov in range(N_cov):
            nc_data_cov = Dataset(lfile_cov[icov],'r')
            print(".... Load "+lid_cov[icov]+" dataset (training data): "+lfile_cov[icov])
            cov.append(Imputing_NaN_3d(np.copy(nc_data_cov[lname_cov[icov]][:,indLat,indLon])))
            nc_data_cov.close()


    ## Apply reduction parameter
    if dwscale!=1:
        x_mod    = einops.reduce(x_mod,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmedian)
        x_obs    = einops.reduce(x_obs,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmedian)
        x_OI     = einops.reduce(x_OI,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmedian)
        for icov in range(N_cov):
            cov[icov]     = einops.reduce(cov[icov],  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmedian)
        if domain=="OSMOSIS":
            indLat     = np.arange(0,int(200/dwscale))
            indLon     = np.arange(0,int(160/dwscale))
        else:
            indLat     = np.arange(0,int(200/dwscale))
            indLon     = np.arange(0,int(200/dwscale))

    # Define mask
    mask  = np.copy(x_obs)
    mask  = np.asarray(~np.isnan(mask))

    # create the time series (additional 4th time dimension)
    input_train     = ndarray_NaN((len(indN_Tr),len(indLat),len(indLon),size_tw))
    mask_train      = np.zeros((len(indN_Tr),len(indLat),len(indLon),size_tw))
    target_train    = ndarray_NaN((len(indN_Tr),len(indLat),len(indLon),size_tw)) 
    x_train_OI      = ndarray_NaN((len(indN_Tr),len(indLat),len(indLon),size_tw))
    if include_covariates==True:
        cov_train      = []
        mask_cov_train = []
        for icov in range(N_cov):
            cov_train.append(ndarray_NaN((len(indN_Tr),len(indLat),len(indLon),size_tw)))
            mask_cov_train.append(np.ones((len(indN_Tr),len(indLat),len(indLon),size_tw)))
    id_rm = []
    for k in range(len(indN_Tr)):
        idt = np.arange(indN_Tr[k]-np.floor(size_tw/2.),indN_Tr[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<x_mod.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        if len(idt)<size_tw:
          id_rm.append(k)
        # build the training datasets
        if flagloadOIData == 1:
            x_train_OI[k,:,:,idt2]   = x_OI[idt,:,:]
            target_train[k,:,:,idt2] = x_mod[idt,:,:] - x_OI[idt,:,:]
            input_train[k,:,:,idt2]  = x_obs[idt,:,:] - x_OI[idt,:,:]
        else:
            target_train[k,:,:,idt2] = x_mod[idt,:,:]
            input_train[k,:,:,idt2]  = x_obs[idt,:,:]
        mask_train[k,:,:,idt2] = mask[idt,:,:]
        # import covariates
        if include_covariates==True:
            for icov in range(N_cov):
                cov_train[icov][k,:,:,idt2] = cov[icov][idt,:,:]
    # Build ground truth data train
    if flagTrWMissingData==2:
        target_train = input_train
    # Add covariates (merge x_train and mask_train with covariates)
    if include_covariates==True:
        cov_train.insert(0,input_train)
        mask_cov_train.insert(0,mask_train)
        input_train     = np.concatenate(cov_train,axis=3)
        mask_train      = np.concatenate(mask_cov_train,axis=3)
        order           = np.stack([np.arange(i*size_tw,(i+1)*size_tw) for i in range(N_cov+1)]).T.flatten()
        input_train     = input_train[:,:,:,order]
        mask_train      = mask_train[:,:,:,order]
    if len(id_rm)>0:
        target_train    = np.delete(target_train,id_rm,axis=0)
        input_train     = np.delete(input_train,id_rm,axis=0)
        mask_train      = np.delete(mask_train,id_rm,axis=0)
        x_train_OI      = np.delete(x_train_OI,id_rm,axis=0)
    print('.... # loaded samples: %d '%input_train.shape[0])

    if flagloadOIData:
        print("....... # of training patches: %d/%d"%(input_train.shape[0],x_train_OI.shape[0]))
    else:
        print("....... # of training patches: %d"%(input_train.shape[0]))
      
    # *** TEST DATASET ***#
    print("2) .... Load SST dataset (test data): "+fileObs)      

    # create the time series (additional 4th time dimension)
    input_test = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    mask_test  = np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw))
    target_test    = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    x_test_OI  = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    if include_covariates==True:
        cov_test      = []
        mask_cov_test = []
        for icov in range(N_cov):
            cov_test.append(ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw)))
            mask_cov_test.append(np.ones((len(indN_Tt),len(indLat),len(indLon),size_tw)))
    id_rm = []
    for k in range(len(indN_Tt)):
        idt = np.arange(indN_Tt[k]-np.floor(size_tw/2.),indN_Tt[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<x_mod.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        # build the testing datasets
        if flagloadOIData == 1:
            x_test_OI[k,:,:,idt2]   = x_OI[idt,:,:]
            target_test[k,:,:,idt2] = x_mod[idt,:,:] - x_OI[idt,:,:]
            input_test[k,:,:,idt2]  = x_obs[idt,:,:] - x_OI[idt,:,:]
        else:
            target_test[k,:,:,idt2] = x_mod[idt,:,:]
            input_test[k,:,:,idt2]  = x_obs[idt,:,:]
        mask_test[k,:,:,idt2] = mask[idt,:,:]
        # import covariates
        if include_covariates==True:
            for icov in range(N_cov):
                cov_test[icov][k,:,:,idt2] = cov[icov][idt,:,:]
    # Build ground truth data test
    if flagTrWMissingData==2:
        target_test = input_test
    # Add covariates (merge x_test and mask_test with covariates)
    if include_covariates==True:
        cov_test.insert(0,input_test)
        mask_cov_test.insert(0,mask_test)
        input_test     = np.concatenate(cov_test,axis=3)
        mask_test      = np.concatenate(mask_cov_test,axis=3)
        order           = np.stack([np.arange(i*size_tw,(i+1)*size_tw) for i in range(N_cov+1)]).T.flatten()
        input_test     = input_test[:,:,:,order]
        mask_test      = mask_test[:,:,:,order]
    if len(id_rm)>0:
        target_test    = np.delete(target_test,id_rm,axis=0)
        input_test     = np.delete(input_test,id_rm,axis=0)
        mask_test      = np.delete(mask_test,id_rm,axis=0)
        x_test_OI      = np.delete(x_test_OI,id_rm,axis=0)
    print('.... # loaded samples: %d '%input_test.shape[0])

    genFilename = 'modelNATL60_SSH_'+str('%03d'%input_train.shape[0])+str('_%03d'%input_train.shape[1])+str('_%03d'%input_train.shape[2])
        
    if include_covariates==False: 
        meanTr          = np.nanmean( target_train )
        stdTr           = np.sqrt( np.nanmean( target_train**2 ) )
        input_train     = (input_train - meanTr)/stdTr
        target_train    = (target_train - meanTr)/stdTr
        input_test      = (input_test - meanTr)/stdTr
        target_test     = (target_test - meanTr)/stdTr
    else:
        index = np.asarray([np.arange(i,(N_cov+1)*size_tw,(N_cov+1)) for i in range(N_cov+1)])
        meanTr          = [np.nanmean(input_train[:,:,:,index[i,:]]) for i in range(N_cov+1)]
        stdTr           = [np.sqrt(np.nanvar(input_train[:,:,:,index[i,:]])) for i in range(N_cov+1)]
        meanTr[0]       = np.nanmean( target_train )
        stdTr[0]        = np.sqrt( np.nanmean( target_train**2 ) )
        for i in range(N_cov+1):
            input_train[:,:,:,index[i]]  = (input_train[:,:,:,index[i]] - meanTr[i])/stdTr[i]
            input_test[:,:,:,index[i]]   = (input_test[:,:,:,index[i]] - meanTr[i])/stdTr[i]
        target_train = (target_train - meanTr[0])/stdTr[0]
        target_test  = (target_test  - meanTr[0])/stdTr[0]

    print("... (after normalization) mean Tr = %f"%(np.nanmean(target_train)))
    print("... (after normalization) mean Tt = %f"%(np.nanmean(target_test)))
      
    return genFilename,\
           input_train,mask_train,target_train,x_train_OI,lday_pred,meanTr,stdTr,\
           input_test,mask_test,target_test,x_test_OI,lday_test
