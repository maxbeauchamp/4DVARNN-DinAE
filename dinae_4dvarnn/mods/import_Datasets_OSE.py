from dinae_4dvarnn import *
from tools import *

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

    # list of test dates
    indN_Tt = range(365)
    lday_train=[ datetime.strftime(datetime.strptime("2017-01-01",'%Y-%m-%d')\
                          + timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tt ]
    if domain=="OSMOSIS":
        indLat     = np.arange(0,200)
        indLon     = np.arange(0,160)         
    else:
        indLat     = np.arange(0,200)
        indLon     = np.arange(0,200)     
 
    ## Load datasets
    print("1) .... Load SSH dataset (training data): "+fileObs)
    nc_data_obs = Dataset(fileObs,'r')  
    inputs      = np.copy(nc_data_obs['ssh'][:,indLat,indLon])
    lags        = np.copy(nc_data_obs['lag'][:,indLat,indLon])
    targets = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon)))

    # load OI data
    if flagloadOIData == 1:
        print(".... Load OI SSH dataset (training data): "+fileOI)
        nc_data    = Dataset(fileOI,'r')
        x_OI = Imputing_NaN_3d(np.copy(nc_data['ssh'][:,indLat,indLon]))
        nc_data.close()
        # remove distribution tail (x_OI-x) 
        anomaly = x_OI-inputs
        anomaly_masked = anomaly[np.asarray(~np.isnan(anomaly))]
        index = np.where( (anomaly<=np.quantile(anomaly_masked,.05)) | \
              (anomaly>=np.quantile(anomaly_masked,.95)) )
        inputs[index] = np.nan

    # Remove data from input
    NPATCH     = 5
    SIZE_PATCH = 10
    for i in range(len(inputs)):
        posx = np.random.randint(min(indLon)+SIZE_PATCH,max(indLon)-SIZE_PATCH,NPATCH)
        posy = np.random.randint(min(indLat)+SIZE_PATCH,max(indLat)-SIZE_PATCH,NPATCH)
        for ipatch in range(NPATCH):
            # define target as patch-collected data
            targets[i,
                   (posx[ipatch]-SIZE_PATCH):(posx[ipatch]+SIZE_PATCH+1),
                   (posy[ipatch]-SIZE_PATCH):(posy[ipatch]+SIZE_PATCH+1)] =\
            inputs[i,
                   (posx[ipatch]-SIZE_PATCH):(posx[ipatch]+SIZE_PATCH+1),
                   (posy[ipatch]-SIZE_PATCH):(posy[ipatch]+SIZE_PATCH+1)]
        for ipatch in range(NPATCH):
            # remove patch-collected data from inputs
            inputs[i,
                   (posx[ipatch]-SIZE_PATCH):(posx[ipatch]+SIZE_PATCH+1),
                   (posy[ipatch]-SIZE_PATCH):(posy[ipatch]+SIZE_PATCH+1)] = np.nan

    # Keep only 0 timelag flag as targets
    index = np.where( (~np.isnan(targets)) & (np.abs(lags)>0.5) )
    targets[index] = np.nan

    # Define mask
    mask_inputs  = np.copy(inputs)
    mask_inputs  = np.asarray(~np.isnan(mask_inputs))
    mask_targets = np.copy(targets)
    mask_targets = np.asarray(~np.isnan(mask_targets))
    sat          = np.copy(nc_data_obs['sat'][:,indLat,indLon])
    time         = np.copy(nc_data_obs.variables.get('Time')[:,indLat,indLon]%86400)
    nc_data_obs.close()

    ## Load covariates
    if include_covariates==True:
        cov=[]
        for icov in range(N_cov):
            nc_data_cov = Dataset(lfile_cov[icov],'r')
            print(".... Load "+lid_cov[icov]+" dataset (training data): "+lfile_cov[icov])
            cov.append(Imputing_NaN_3d(np.copy(nc_data_cov[lname_cov[icov]][:,indLat,indLon])))
            nc_data_cov.close()

    ## Build time series for training (Ntime * Nchannel * Nlat * Nlon)
    inputs_train      = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    mask_inputs_train = np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw))
    target_train      = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw)) 
    mask_targets_train = np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw))
    sat_train         = np.empty((len(indN_Tt),len(indLat),len(indLon),size_tw),dtype=object)
    sat_train.fill('')
    time_train        = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    inputs_train_OI   = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    if include_covariates==True:
        cov_train      = []
        mask_cov_train = []
        for icov in range(N_cov):
            cov_train.append(ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw)))
            mask_cov_train.append(np.ones((len(indN_Tt),len(indLat),len(indLon),size_tw)))
    for k in range(len(indN_Tt)):
        idt = np.arange(indN_Tt[k]-np.floor(size_tw/2.),indN_Tt[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<x_OI.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        if flagloadOIData == 1: 
            inputs_train_OI[k,:,:,idt2]      = x_OI[idt,:,:]
            inputs_train[k,:,:,idt2]         = inputs[idt,:,:] - x_OI[idt,:,:]
            target_train[k,:,:,idt2]         = targets[idt,:,:] - x_OI[idt,:,:]
        else:
            inputs_train[k,:,:,idt2]    = inputs[idt,:,:]
            target_train[k,:,:,idt2]    = targets[idt,:,:]
        sat_train[k,:,:,idt2]    = sat[idt,:,:]
        time_train[k,:,:,idt2]   = time[idt,:,:]
        # import covariates
        if include_covariates==True:
            for icov in range(N_cov):
                cov_train[icov][k,:,:,idt2] = cov[icov][idt,:,:]
        mask_inputs_train[k,:,:,idt2] = mask_inputs[idt,:,:]
        mask_targets_train[k,:,:,idt2] = mask_targets[idt,:,:]
        
    ## Add covariates (merge inputs_train and mask_train with covariates)
    if include_covariates==True:
        cov_train.insert(0,inputs_train)
        mask_cov_train.insert(0,mask_inputs_train)
        inputs_train    = np.concatenate(cov_train,axis=3)
        mask_inputs_train = np.concatenate(mask_cov_train,axis=3)
        order     = np.stack([np.arange(i*size_tw,(i+1)*size_tw) for i in range(0,N_cov+1)]).T.flatten()
        inputs_train    = inputs_train[:,:,:,order]
        mask_inputs_train = mask_inputs_train[:,:,:,order]

    ## Build gappy (and potentially noisy) data test
    inputs_train_missing = (inputs_train * mask_inputs_train)
    genFilename = 'modelNATL60_SSH_'+str('%03d'%inputs_train.shape[0])+str('_%03d'%inputs_train.shape[1])+str('_%03d'%inputs_train.shape[2])
    print('....... Generic model filename: '+genFilename)
    
    ## Normalization
    if include_covariates==False:
        meanTt               = np.nanmean( inputs_train )
        stdTt                = np.sqrt( np.nanvar(inputs_train) )
        inputs_train          = (inputs_train  - meanTt)/stdTt
        inputs_train_missing  = (inputs_train_missing - meanTt)/stdTt
        target_train          = (target_train - meanTt)/stdTt
    else:
        index = np.asarray([np.arange(i,(N_cov+1)*size_tw,(N_cov+1)) for i in range(N_cov+1)])
        meanTt               = [np.nanmean(inputs_train[:,:,:,index[i,:]]) for i in range(N_cov+1)]
        stdTt                = [np.sqrt(np.nanvar(inputs_train[:,:,:,index[i,:]])) for i in range(N_cov+1)]
        for i in range(N_cov+1):
            inputs_train[:,:,:,index[i]]          = (inputs_train[:,:,:,index[i]] - meanTt[i])/stdTt[i]
            inputs_train_missing[:,:,:,index[i]]  = (inputs_train_missing[:,:,:,index[i]] - meanTt[i])/stdTt[i]
        target_train  = (target_train - meanTt[0])/stdTt[0]

    ## Hanning
    EdgeWidth  = 4
    EdgeWindow = np.zeros((inputs_train.shape[1],inputs_train.shape[2]))
    EdgeWindow[EdgeWidth:inputs_train.shape[1]-EdgeWidth,EdgeWidth:inputs_train.shape[2]-EdgeWidth] = 1
    inputs_train  = np.moveaxis(np.moveaxis(inputs_train,3,1) * np.tile(EdgeWindow,(inputs_train.shape[0],inputs_train.shape[3],1,1)),1,3)
    target_train  = np.moveaxis(np.moveaxis(target_train,3,1) * np.tile(EdgeWindow,(target_train.shape[0],target_train.shape[3],1,1)),1,3)
    inputs_train_missing = np.moveaxis(np.moveaxis(inputs_train_missing,3,1) * np.tile(EdgeWindow,(inputs_train_missing.shape[0],inputs_train_missing.shape[3],1,1)),1,3)
    mask_inputs_train  = np.moveaxis(np.moveaxis(mask_inputs_train,3,1) * np.tile(EdgeWindow,(mask_inputs_train.shape[0],inputs_train.shape[3],1,1)),1,3)
    mask_targets_train  = np.moveaxis(np.moveaxis(mask_targets_train,3,1) * np.tile(EdgeWindow,(mask_targets_train.shape[0],mask_targets_train.shape[3],1,1)),1,3)
    print(".... Test set shape     %dx%dx%dx%d"%(inputs_train.shape[0],inputs_train.shape[1],inputs_train.shape[2],inputs_train.shape[3]))

    ## Export datatest as NetCDF
    export_NetCDF=False
    if export_NetCDF==True:
        dt64 = [ np.datetime64(datetime.strptime("2017-01-01",'%Y-%m-%d')\
                                    + timedelta(days=np.float64(i))) for i in indN_Tt ]
        time_u = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        mesh_lat, mesh_lon = np.meshgrid(indLat, indLon)
        data = xr.Dataset(\
                        data_vars={'longitude': (('lat','lon'),mesh_lon),\
                                   'latitude' : (('lat','lon'),mesh_lat),\
                                   'Time'     : (('time'),time_u),\
                                   'inputs'   : (('time','nc','lat','lon'),inputs_train[:,:,:,index[0]].transpose(0,3,1,2)),\
                                   'mask_inputs'  : (('time','nc','lat','lon'),mask_inputs_train[:,:,:,index[0]].transpose(0,3,1,2)),\
                                   'target'   : (('time','nc','lat','lon'),target_train.transpose(0,3,1,2)),\
                                   'mask_targets' : (('time','nc','lat','lon'),mask_targets_train.transpose(0,3,1,2)),\
                                   'sat'      : (('time','nc','lat','lon'),sat_train.transpose(0,3,1,2)),\
                                   'nadir_time'     : (('time','nc','lat','lon'),time_train.transpose(0,3,1,2))},\
                        coords={'lon': indLon,\
                                'lat': indLat,\
                                'time': range(0,len(time_u))})
        data.to_netcdf(dirSAVE+"data_train.nc")

    return genFilename, meanTt, stdTt, inputs_train, mask_inputs_train, target_train, mask_targets_train, inputs_train_missing, sat_train, time_train, lday_train, inputs_train_OI


