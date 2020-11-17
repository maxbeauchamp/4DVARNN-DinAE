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
    thrMisData = 0.005
    # list of test dates
    indN_Tt = np.concatenate([np.arange(60,80),np.arange(140,160),\
                             np.arange(220,240),np.arange(300,320)])
    indN_Tr = np.delete(range(365),indN_Tt)
    indN_Tt = np.arange(start_eval_index,end_eval_index)   # index of evaluation period
    indN_Tr = np.arange(start_train_index,end_train_index) # index of training period
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
    x_orig      = Imputing_NaN_3d(np.copy(nc_data_mod['ssh'][:,indLat,indLon]))
    # masking strategy differs according to flagTrWMissingData flag 
    mask_orig         = np.copy(nc_data_obs['ssh_mod'][:,indLat,indLon])
    mask_orig         = np.asarray(~np.isnan(mask_orig))
    if flagTrWMissingData==0:
        mask_orig[indN_Tr,:,:]  = 1
    if type_obs=="obs":
        noisy_obs = np.copy(nc_data_obs['ssh_obs'][:,indLat,indLon])
        obs       = np.copy(nc_data_obs['ssh_mod'][:,indLat,indLon])
        err_orig  = noisy_obs - obs
        err_orig = np.where(np.isnan(err_orig), 0, err_orig)
    nc_data_mod.close()
    nc_data_obs.close()
    # load OI data
    if flagloadOIData == 1:
        print(".... Load OI SSH dataset (training data): "+fileOI)
        nc_data    = Dataset(fileOI,'r')
        x_OI = Imputing_NaN_3d(np.copy(nc_data['ssh_mod'][:,indLat,indLon]))
        nc_data.close()
    # load covariates
    if include_covariates==True:
        cov=[]
        for icov in range(N_cov):
            nc_data_cov = Dataset(lfile_cov[icov],'r')
            print(".... Load "+lid_cov[icov]+" dataset (training data): "+lfile_cov[icov])
            cov.append(Imputing_NaN_3d(np.copy(nc_data_cov[lname_cov[icov]][:,indLat,indLon])))
            nc_data_cov.close()

    # create the time series (additional 4th time dimension)
    x_train    = ndarray_NaN((len(indN_Tr),len(indLat),len(indLon),size_tw))
    mask_train = np.zeros((len(indN_Tr),len(indLat),len(indLon),size_tw))
    err_train  = np.zeros((len(indN_Tr),len(indLat),len(indLon),size_tw))
    x_train_OI = ndarray_NaN((len(indN_Tr),len(indLat),len(indLon),size_tw))
    if include_covariates==True:
        cov_train      = []
        mask_cov_train = []
        err_cov_train  = []
        for icov in range(N_cov):
            cov_train.append(ndarray_NaN((len(indN_Tr),len(indLat),len(indLon),size_tw)))
            mask_cov_train.append(np.ones((len(indN_Tr),len(indLat),len(indLon),size_tw)))
            err_cov_train.append(np.zeros((len(indN_Tr),len(indLat),len(indLon),size_tw)))
    id_rm = []
    for k in range(len(indN_Tr)):
        idt = np.arange(indN_Tr[k]-np.floor(size_tw/2.),indN_Tr[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<x_orig.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        if len(idt)<size_tw:
          id_rm.append(k)
        # build the training datasets
        if flagloadOIData == 1:
            x_train_OI[k,:,:,idt2] = x_OI[idt,:,:]
            x_train[k,:,:,idt2]    = x_orig[idt,:,:] - x_OI[idt,:,:]
        else:
            x_train[k,:,:,idt2]    = x_orig[idt,:,:]
        if type_obs=="obs":
            err_train[k,:,:,idt2] = err_orig[idt,:,:]
        # import covariates
        if include_covariates==True:
            for icov in range(N_cov):
                cov_train[icov][k,:,:,idt2] = cov[icov][idt,:,:]
        mask_train[k,:,:,idt2] = mask_orig[idt,:,:]
    # Build ground truth data train
    if flagTrWMissingData==2:
        gt_train = (x_train * mask_train) + err_train
    else:
        gt_train = x_train
    # Add covariates (merge x_train and mask_train with covariates)
    if include_covariates==True:
        cov_train.insert(0,x_train)
        mask_cov_train.insert(0,mask_train)
        err_cov_train.insert(0,err_train)
        x_train    = np.concatenate(cov_train,axis=3)
        mask_train = np.concatenate(mask_cov_train,axis=3)
        err_train  = np.concatenate(err_cov_train,axis=3)
        order      = np.stack([np.arange(i*size_tw,(i+1)*size_tw) for i in range(N_cov+1)]).T.flatten()
        x_train    = x_train[:,:,:,order]
        mask_train = mask_train[:,:,:,order]
        err_train  = err_train[:,:,:,order]
    # Build gappy (and potentially noisy) data train
    x_train_missing = (x_train * mask_train) + err_train
    if len(id_rm)>0:
        gt_train        = np.delete(gt_train,id_rm,axis=0)
        x_train         = np.delete(x_train,id_rm,axis=0)
        x_train_missing = np.delete(x_train_missing,id_rm,axis=0)
        mask_train      = np.delete(mask_train,id_rm,axis=0)
        x_train_OI      = np.delete(x_train_OI,id_rm,axis=0)
    print('.... # loaded samples: %d '%x_train.shape[0])

    # remove patch if no SSH data
    ss              = np.sum( np.sum( np.sum( x_train < -100 , axis = -1) , axis = -1 ) , axis = -1)
    ind             = np.where( ss == 0 )
    x_train         = x_train[ind[0],:,:,:]
    gt_train        = gt_train[ind[0],:,:,:]
    x_train_missing = x_train_missing[ind[0],:,:,:]
    mask_train      = mask_train[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_train_OI  = x_train_OI[ind[0],:,:,:]
    rateMissDataTr_ = np.asarray(np.sum( np.sum( np.sum( mask_train , axis = -1) , axis = -1 ) , axis = -1), dtype=np.float64)
    rateMissDataTr_ /= mask_train.shape[1]*mask_train.shape[2]*mask_train.shape[3]
    ind             = np.where( rateMissDataTr_  >= thrMisData )
    gt_train        = gt_train[ind[0],:,:,:]
    x_train         = x_train[ind[0],:,:,:]
    x_train_missing = x_train_missing[ind[0],:,:,:]
    mask_train      = mask_train[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_train_OI = x_train_OI[ind[0],:,:,:]
    y_train = np.ones((x_train.shape[0]))

    if flagloadOIData:
        print("....... # of training patches: %d/%d"%(x_train.shape[0],x_train_OI.shape[0]))
    else:
        print("....... # of training patches: %d"%(x_train.shape[0]))
      
    # *** TEST DATASET ***#
    print("2) .... Load SST dataset (test data): "+fileObs)      

    # create the time series (additional 4th time dimension)
    x_test    = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    mask_test = np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw))
    err_test  = np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw))
    x_test_OI = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    if include_covariates==True:
        cov_test      = []
        mask_cov_test = []
        err_cov_test  = []
        for icov in range(N_cov):
            cov_test.append(ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw)))
            mask_cov_test.append(np.ones((len(indN_Tt),len(indLat),len(indLon),size_tw)))
            err_cov_test.append(np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw)))
    for k in range(len(indN_Tt)):
        idt = np.arange(indN_Tt[k]-np.floor(size_tw/2.),indN_Tt[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<x_orig.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        if flagloadOIData == 1: 
            x_test_OI[k,:,:,idt2] = x_OI[idt,:,:]
            x_test[k,:,:,idt2]    = x_orig[idt,:,:] - x_OI[idt,:,:]
        else:
            x_test[k,:,:,idt2]    = x_orig[idt,:,:]
        if type_obs=="obs":
            err_test[k,:,:,idt2] = err_orig[idt,:,:]
        # import covariates
        if include_covariates==True:
            for icov in range(N_cov):
                cov_test[icov][k,:,:,idt2] = cov[icov][idt,:,:]
        mask_test[k,:,:,idt2] = mask_orig[idt,:,:]
    # Build ground truth data test
    gt_test = x_test
    # Add covariates (merge x_test and mask_test with covariates)
    if include_covariates==True:
        cov_test.insert(0,x_test)
        mask_cov_test.insert(0,mask_test)
        err_cov_test.insert(0,err_test)
        x_test    = np.concatenate(cov_test,axis=3)
        mask_test = np.concatenate(mask_cov_test,axis=3)
        err_test  = np.concatenate(err_cov_test,axis=3)
        order      = np.stack([np.arange(i*size_tw,(i+1)*size_tw) for i in range(0,N_cov+1)]).T.flatten()
        x_test    = x_test[:,:,:,order]
        mask_test = mask_test[:,:,:,order]
        err_test  = err_test[:,:,:,order]
    # Build gappy (and potentially noisy) data test
    x_test_missing = (x_test * mask_test) + err_test
    print('.... # loaded samples: %d '%x_test.shape[0])

    # remove patch if no SSH data
    ss            = np.sum( np.sum( np.sum( x_test < -100 , axis = -1) , axis = -1 ) , axis = -1)
    ind           = np.where( ss == 0 )
    x_test         = x_test[ind[0],:,:,:]
    gt_test        = gt_test[ind[0],:,:,:]
    x_test_missing = x_test_missing[ind[0],:,:,:]
    mask_test      = mask_test[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_test_OI = x_test_OI[ind[0],:,:,:]
    rateMissDataTr_ = np.asarray(np.sum( np.sum( np.sum( mask_test , axis = -1) , axis = -1 ) , axis = -1), dtype=np.float64)
    rateMissDataTr_ /= mask_test.shape[1]*mask_test.shape[2]*mask_test.shape[3]
    ind        = np.where( rateMissDataTr_  >= thrMisData )
    x_test         = x_test[ind[0],:,:,:]
    gt_test        = gt_test[ind[0],:,:,:]
    x_test_missing = x_test_missing[ind[0],:,:,:]
    mask_test      = mask_test[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_test_OI = x_test_OI[ind[0],:,:,:]

    y_test    = np.ones((x_test.shape[0]))

    if flagloadOIData:
        print("....... # of test patches: %d /%d"%(x_test.shape[0],x_test_OI.shape[0]))
    else:
        print("....... # of test patches: %d"%(x_test.shape[0]))

    print("... mean Tr = %f"%(np.mean(gt_train)))
    print("... mean Tt = %f"%(np.mean(gt_test)))
            
    print(".... Training set shape %dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
    print(".... Test set shape     %dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
    
    genFilename = 'modelNATL60_SSH_'+str('%03d'%x_train.shape[0])+str('_%03d'%x_train.shape[1])+str('_%03d'%x_train.shape[2])
        
    print('....... Generic model filename: '+genFilename)
    
    if include_covariates==False: 
        meanTr          = np.mean( x_train )
        stdTr           = np.sqrt( np.mean( x_train**2 ) )
        x_train         = (x_train - meanTr)/stdTr
        x_train_missing = (x_train_missing - meanTr)/stdTr
        gt_train        = (gt_train - meanTr)/stdTr
        x_test          = (x_test  - meanTr)/stdTr
        x_test_missing  = (x_test_missing - meanTr)/stdTr
        gt_test         = (gt_test - meanTr)/stdTr
    else:
        index = np.asarray([np.arange(i,(N_cov+1)*size_tw,(N_cov+1)) for i in range(N_cov+1)])
        meanTr          = [np.mean(x_train[:,:,:,index[i,:]]) for i in range(N_cov+1)]
        stdTr           = [np.sqrt(np.var(x_train[:,:,:,index[i,:]])) for i in range(N_cov+1)]
        for i in range(N_cov+1):
            x_train[:,:,:,index[i]]         = (x_train[:,:,:,index[i]] - meanTr[i])/stdTr[i]
            x_train_missing[:,:,:,index[i]] = (x_train_missing[:,:,:,index[i]] - meanTr[i])/stdTr[i]
            x_test[:,:,:,index[i]]          = (x_test[:,:,:,index[i]] - meanTr[i])/stdTr[i]
            x_test_missing[:,:,:,index[i]]  = (x_test_missing[:,:,:,index[i]] - meanTr[i])/stdTr[i]
        gt_train = (gt_train - meanTr[0])/stdTr[0]
        gt_test  = (gt_test - meanTr[0])/stdTr[0]
    #print('... Mean and std of training data: %f  -- %f'%tuple(meanTr,stdTr))

    if flagDataWindowing == 1:
        HannWindow = np.reshape(np.hanning(x_train.shape[2]),(x_train.shape[1],1)) * np.reshape(np.hanning(x_train.shape[1]),(x_train.shape[2],1)).transpose() 

        x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(HannWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
        gt_train = np.moveaxis(np.moveaxis(gt_train,3,1) * np.tile(HannWindow,(gt_train.shape[0],gt_train.shape[3],1,1)),1,3)
        x_train_missing = np.moveaxis(np.moveaxis(x_train_missing,3,1) * np.tile(HannWindow,(x_train_missing.shape[0],x_train_missing.shape[3],1,1)),1,3)

        x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(HannWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
        gt_test  = np.moveaxis(np.moveaxis(gt_test,3,1) * np.tile(HannWindow,(gt_test.shape[0],gt_test.shape[3],1,1)),1,3)
        x_test_missing = np.moveaxis(np.moveaxis(x_test_missing,3,1) * np.tile(HannWindow,(x_test_missing.shape[0],x_test_missing.shape[3],1,1)),1,3)

        print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

    elif flagDataWindowing == 2:
        EdgeWidth  = 4
        EdgeWindow = np.zeros((x_train.shape[1],x_train.shape[2]))
        EdgeWindow[EdgeWidth:x_train.shape[1]-EdgeWidth,EdgeWidth:x_train.shape[2]-EdgeWidth] = 1
        
        x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(EdgeWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
        gt_train = np.moveaxis(np.moveaxis(gt_train,3,1) * np.tile(EdgeWindow,(gt_train.shape[0],gt_train.shape[3],1,1)),1,3)
        x_train_missing = np.moveaxis(np.moveaxis(x_train_missing,3,1) * np.tile(EdgeWindow,(x_train_missing.shape[0],x_train_missing.shape[3],1,1)),1,3)

        x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(EdgeWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
        gt_test  = np.moveaxis(np.moveaxis(gt_test,3,1) * np.tile(EdgeWindow,(gt_test.shape[0],gt_test.shape[3],1,1)),1,3)
        x_test_missing = np.moveaxis(np.moveaxis(x_test_missing,3,1) * np.tile(EdgeWindow,(x_test_missing.shape[0],x_test_missing.shape[3],1,1)),1,3)

        mask_train = np.moveaxis(np.moveaxis(mask_train,3,1) * np.tile(EdgeWindow,(mask_train.shape[0],x_train.shape[3],1,1)),1 ,3)
        mask_test  = np.moveaxis(np.moveaxis(mask_test,3,1) * np.tile(EdgeWindow,(mask_test.shape[0],x_test.shape[3],1,1)),1,3)
        print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
      
    print("... (after normalization) mean Tr = %f"%(np.mean(gt_train)))
    print("... (after normalization) mean Tt = %f"%(np.mean(gt_test)))
      
    return genFilename,\
           x_train,y_train,mask_train,gt_train,x_train_missing,x_train_OI,lday_pred,meanTr,stdTr,\
           x_test,y_test,mask_test,gt_test,x_test_missing,x_test_OI,lday_test
