from 4dvarnn-dinae import *

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

def import_Data_OSE(dict_global_Params):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    #*** Start reading the data ***#
    thrMisData = 0.005
    # list of test dates
    indN_Tt = range(365)
    lday_test=[ datetime.strftime(datetime.strptime("2017-01-01",'%Y-%m-%d')\
                          + timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tt ]

    if domain=="OSMOSIS":
        indLat     = np.arange(0,200)
        indLon     = np.arange(0,160)         
    else:
        indLat     = np.arange(0,200)
        indLon     = np.arange(0,200)     
 
    print("1) .... Load SSH dataset (training data): "+fileObs)
    nc_data_obs = Dataset(fileObs,'r')  
    x_orig      = np.copy(nc_data_obs['ssh'][:,indLat,indLon])
    # masking strategie differs according to flagTrWMissingData flag 
    mask_orig         = np.copy(x_orig)
    mask_orig         = np.asarray(~np.isnan(mask_orig))
    nc_data_obs.close()
    # load OI data
    if flagloadOIData == 1:
        print(".... Load OI SSH dataset (training data): "+fileOI)
        nc_data    = Dataset(fileOI,'r')
        x_OI = Imputing_NaN_3d(np.copy(nc_data['ssh'][:,indLat,indLon]))
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
    x_test    = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    mask_test = np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw))
    x_test_OI = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    if include_covariates==True:
        cov_test      = []
        mask_cov_test = []
        for icov in range(N_cov):
            cov_test.append(ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw)))
            mask_cov_test.append(np.ones((len(indN_Tt),len(indLat),len(indLon),size_tw)))
    for k in range(len(indN_Tt)):
        idt = np.arange(indN_Tt[k]-np.floor(size_tw/2.),indN_Tt[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<x_OI.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        if flagloadOIData == 1: 
            x_test_OI[k,:,:,idt2] = x_OI[idt,:,:]
            x_test[k,:,:,idt2]    = x_orig[idt,:,:] - x_OI[idt,:,:]
        else:
            x_test[k,:,:,idt2]    = x_orig[idt,:,:]
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
        x_test    = np.concatenate(cov_test,axis=3)
        mask_test = np.concatenate(mask_cov_test,axis=3)
        order      = np.stack([np.arange(i*size_tw,(i+1)*size_tw) for i in range(0,N_cov+1)]).T.flatten()
        x_test    = x_test[:,:,:,order]
        mask_test = mask_test[:,:,:,order]
    # Build gappy (and potentially noisy) data test
    x_test_missing = (x_test * mask_test)
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

    print("... mean Tt = %f"%(np.nanmean(gt_test)))
            
    print(".... Test set shape     %dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
    
    genFilename = 'modelNATL60_SSH_'+str('%03d'%x_test.shape[0])+str('_%03d'%x_test.shape[1])+str('_%03d'%x_test.shape[2])
        
    print('....... Generic model filename: '+genFilename)
    
    if include_covariates==False:
        meanTt          = np.nanmean( x_test )
        stdTt           = np.sqrt( np.nanvar(x_test) )
        x_test          = (x_test  - meanTt)/stdTt
        x_test_missing  = (x_test_missing - meanTt)/stdTt
        gt_test         = (gt_test - meanTt)/stdTt
    else:
        index = np.asarray([np.arange(i,(N_cov+1)*size_tw,(N_cov+1)) for i in range(N_cov+1)])
        meanTt          = [np.nanmean(x_test[:,:,:,index[i,:]]) for i in range(N_cov+1)]
        stdTt           = [np.sqrt(np.nanvar(x_test[:,:,:,index[i,:]])) for i in range(N_cov+1)]
        for i in range(N_cov+1):
            x_test[:,:,:,index[i]]          = (x_test[:,:,:,index[i]] - meanTt[i])/stdTt[i]
            x_test_missing[:,:,:,index[i]]  = (x_test_missing[:,:,:,index[i]] - meanTt[i])/stdTt[i]
        gt_test  = (gt_test - meanTt[0])/stdTt[0]

    if flagDataWindowing == 1:
        HannWindow = np.reshape(np.hanning(x_test.shape[2]),(x_test.shape[1],1)) * np.reshape(np.hanning(x_test.shape[1]),(x_test.shape[2],1)).transpose() 
        x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(HannWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
        gt_test  = np.moveaxis(np.moveaxis(gt_test,3,1) * np.tile(HannWindow,(gt_test.shape[0],gt_test.shape[3],1,1)),1,3)
        x_test_missing = np.moveaxis(np.moveaxis(x_test_missing,3,1) * np.tile(HannWindow,(x_test_missing.shape[0],x_test_missing.shape[3],1,1)),1,3)
        print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

    elif flagDataWindowing == 2:
        EdgeWidth  = 4
        EdgeWindow = np.zeros((x_test.shape[1],x_test.shape[2]))
        EdgeWindow[EdgeWidth:x_test.shape[1]-EdgeWidth,EdgeWidth:x_test.shape[2]-EdgeWidth] = 1
        x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(EdgeWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
        gt_test  = np.moveaxis(np.moveaxis(gt_test,3,1) * np.tile(EdgeWindow,(gt_test.shape[0],gt_test.shape[3],1,1)),1,3)
        x_test_missing = np.moveaxis(np.moveaxis(x_test_missing,3,1) * np.tile(EdgeWindow,(x_test_missing.shape[0],x_test_missing.shape[3],1,1)),1,3)
        mask_test  = np.moveaxis(np.moveaxis(mask_test,3,1) * np.tile(EdgeWindow,(mask_test.shape[0],x_test.shape[3],1,1)),1,3)
        print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
      
    print("... (after normalization) mean Tt = %f"%(np.nanmean(gt_test)))
      
    return genFilename, meanTt, stdTt, x_test, y_test, mask_test, gt_test, x_test_missing, lday_test, x_test_OI

