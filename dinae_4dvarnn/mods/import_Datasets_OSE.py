from dinae_4dvarnn import *
from tools import *
from scipy import ndimage

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
    print(".... Load SSH dataset (training data): "+fileObs)
    nc_data_obs = Dataset(fileObs,'r')  
    inputs      = np.copy(nc_data_obs['ssh'][:,indLat,indLon])
    #inputs      = np.copy(nc_data_obs['ssh_filtered_N5'][:,indLat,indLon])
    lags        = np.copy(nc_data_obs['lag'][:,indLat,indLon])
    targets = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon)))
    nc_data_obs.close()

    # upsample original datasets
    '''for i in range(len(inputs)):
        index_nan = np.where(np.isnan(inputs[i]))
        inputs[i] = ndimage.filters.generic_filter(inputs[i],np.nanmean,size=4)
        inputs[i,index_nan[0],index_nan[1]] = np.nan
    '''

    # load OI data
    print(".... Load OI SSH dataset (training data): "+fileOI)
    nc_data_OI    = Dataset(fileOI,'r')
    x_OI = Imputing_NaN_3d(np.copy(nc_data_OI['ssh'][:,indLat,indLon]))
    if flagloadOIData == 1:
        # remove distribution tail (x_OI-x) 
        anomaly = x_OI-inputs
        anomaly_masked = anomaly[np.asarray(~np.isnan(anomaly))]
        index = np.where( (anomaly<=np.quantile(anomaly_masked,.01)) | 
              (anomaly>=np.quantile(anomaly_masked,.99)) )
        inputs[index] = np.nan
    nc_data_OI.close()

    # load BFN data
    nc_data_BFN   = Dataset(fileBFN,'r')
    x_BFN = Imputing_NaN_3d(np.copy(nc_data_BFN['SSH'][:,:,:]))
    nc_data_BFN.close()

    # load NATL60 data
    print(".... Load NATL60 SSH dataset (training data): "+fileMod)
    nc_data_mod = Dataset(fileMod,'r')
    x_mod       = Imputing_NaN_3d(np.copy(nc_data_mod['ssh'][:,indLat,indLon]))
    print(".... Load NATL60 OI SSH dataset (training data): "+fileOI_Mod)
    nc_data_OI_mod    = Dataset(fileOI_Mod,'r')
    x_OI_mod = Imputing_NaN_3d(np.copy(nc_data_OI_mod['ssh_mod'][:,indLat,indLon]))
    nc_data_OI_mod.close()
    nc_data_mod.close()

    # Remove data from input
    NPATCH     = 30
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

    # test with BFN as targets
    # targets = x_BFN

    ## Load covariates
    if include_covariates==True:
        cov=[]
        cov_mod=[]
        for icov in range(N_cov):
            nc_data_cov = Dataset(lfile_cov[icov],'r')
            print(".... Load "+lid_cov[icov]+" dataset (training data): "+lfile_cov[icov])
            cov.append(Imputing_NaN_3d(np.copy(nc_data_cov[lname_cov[icov]][:,indLat,indLon])))
            nc_data_cov.close()

            nc_data_cov_mod = Dataset(lfile_cov_mod[icov],'r')
            print(".... Load "+lid_cov_mod[icov]+" dataset (training data): "+lfile_cov_mod[icov])
            cov_mod.append(Imputing_NaN_3d(np.copy(nc_data_cov_mod[lname_cov_mod[icov]][:,indLat,indLon])))
            nc_data_cov_mod.close()

    ## Apply reduction parameter
    if dwscale!=1:
        inputs    = einops.reduce(inputs,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmean)
        targets   = einops.reduce(targets,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=int(dwscale/1), w1=int(dwscale/1), reduction=np.nanmean)
        x_OI      = einops.reduce(x_OI,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmean)
        x_mod     = einops.reduce(x_mod,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmean)
        x_OI_mod  = einops.reduce(x_OI_mod,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmean)
        x_BFN     = einops.reduce(x_BFN,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=int(dwscale/2), w1=int(dwscale/2), reduction=np.nanmean)
        for icov in range(N_cov):
            cov[icov]     = einops.reduce(cov[icov],  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmean)
            cov_mod[icov] = einops.reduce(cov_mod[icov],  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=dwscale, w1=dwscale, reduction=np.nanmean)
        if domain=="OSMOSIS":
            indLat     = np.arange(0,int(200/dwscale))
            indLon     = np.arange(0,int(160/dwscale))
        else:
            indLat     = np.arange(0,int(200/dwscale))
            indLon     = np.arange(0,int(200/dwscale))

    # Define mask
    mask_inputs  = np.copy(inputs)
    mask_inputs  = np.asarray(~np.isnan(mask_inputs))
    mask_targets = np.copy(targets)
    mask_targets = np.asarray(~np.isnan(mask_targets))

    # upsample original datasets
    print(np.nanstd(inputs.flatten()-x_OI.flatten()))
    '''index_nan = np.where(~np.isnan(inputs))
    plt.scatter(inputs[index_nan[0],index_nan[1],index_nan[2]].flatten(),
                x_OI[index_nan[0],index_nan[1],index_nan[2]].flatten())
    plt.show()
    '''

    ## Build time series for training (Ntime * Nchannel * Nlat * Nlon)
    inputs_train       = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    targets_train      = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw)) 
    targets_mod         = np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw))
    targets_BFN         = np.zeros((len(indN_Tt),len(indLat),len(indLon),size_tw))
    inputs_train_OI    = ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw))
    if include_covariates==True:
        cov_train      = []
        cov_train_mod  = []
        for icov in range(N_cov):
            cov_train.append(ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw)))
            cov_train_mod.append(ndarray_NaN((len(indN_Tt),len(indLat),len(indLon),size_tw)))
    for k in range(len(indN_Tt)):
        idt = np.arange(indN_Tt[k]-np.floor(size_tw/2.),indN_Tt[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<inputs.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        inputs_train_OI[k,:,:,idt2]          = x_OI[idt,:,:]
        if flagloadOIData == 1: 
            inputs_train[k,:,:,idt2]         = inputs[idt,:,:] - x_OI[idt,:,:]
            targets_train[k,:,:,idt2]        = targets[idt,:,:] - x_OI[idt,:,:]
            targets_mod[k,:,:,idt2]           = x_mod[idt,:,:] - x_OI_mod[idt,:,:] 
            targets_BFN[k,:,:,idt2]           = x_BFN[idt,:,:] - x_OI[idt,:,:]
        else:
            inputs_train[k,:,:,idt2]     = inputs[idt,:,:]
            targets_train[k,:,:,idt2]    = targets[idt,:,:]
            targets_mod[k,:,:,idt2]       = x_mod[idt,:,:]
            targets_BFN[k,:,:,idt2]       = x_BFN[idt,:,:]

        # import covariates
        if include_covariates==True:
            for icov in range(N_cov):
                print(icov)
                cov_train[icov][k,:,:,idt2] = cov[icov][idt,:,:]
                cov_train_mod[icov][k,:,:,idt2] = cov_mod[icov][idt,:,:]
    
    ## Add covariates (merge inputs_train and mask_train with covariates)
    if include_covariates==True:
        cov_train.insert(0,inputs_train)
        cov_train_mod.insert(0,targets_mod)
        inputs_train      = np.concatenate(cov_train,axis=3)
        targets_mod        = np.concatenate(cov_train_mod,axis=3)
        order             = np.stack([np.arange(i*size_tw,(i+1)*size_tw) for i in range(0,N_cov+1)]).T.flatten()
        inputs_train      = inputs_train[:,:,:,order]
        targets_mod        = targets_mod[:,:,:,order]

    ## mask generation
    # mask inputs
    mask_inputs_train   = np.asarray(~np.isnan(inputs_train))
    # mask targets
    mask_targets_train  = np.asarray(~np.isnan(targets_train))
    # mask NATL60
    mask_targets_mod     = np.asarray(~np.isnan(targets_mod))

    ## Build gappy (and potentially noisy) data test
    genFilename = 'modelNATL60_SSH_'+str('%03d'%inputs_train.shape[0])+str('_%03d'%inputs_train.shape[1])+str('_%03d'%inputs_train.shape[2])

    ## Normalization
    if include_covariates==False:
        meanTt               = np.nanmean( inputs_train )
        stdTt                = np.sqrt( np.nanvar(inputs_train) )
        inputs_train         = (inputs_train  - meanTt)/stdTt
        targets_train        = (targets_train - meanTt)/stdTt
        targets_BFN           = (targets_BFN - meanTt)/stdTt

        meanTt_mod            = np.nanmean( targets_mod )
        stdTt_mod             = np.sqrt( np.nanvar(targets_mod) )
        targets_mod            = (targets_mod - meanTt_mod)/stdTt_mod

    else:
        index = np.asarray([np.arange(i,(N_cov+1)*size_tw,(N_cov+1)) for i in range(N_cov+1)])
        meanTt               = [np.nanmean(inputs_train[:,:,:,index[i,:]]) for i in range(N_cov+1)]
        stdTt                = [np.sqrt(np.nanvar(inputs_train[:,:,:,index[i,:]])) for i in range(N_cov+1)]
        meanTt_mod           = [np.nanmean(targets_mod[:,:,:,index[i,:]]) for i in range(N_cov+1)]
        stdTt_mod            = [np.sqrt(np.nanvar(targets_mod[:,:,:,index[i,:]])) for i in range(N_cov+1)]
        for i in range(N_cov+1):
            inputs_train[:,:,:,index[i]]          = (inputs_train[:,:,:,index[i]] - meanTt[i])/stdTt[i]
            targets_mod[:,:,:,index[i]]            = (targets_mod[:,:,:,index[i]] - meanTt_mod[i])/stdTt_mod[i]
        targets_train  = (targets_train - meanTt[0])/stdTt[0]
        targets_BFN  = (targets_BFN - meanTt[0])/stdTt[0]

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
                                   'target'   : (('time','nc','lat','lon'),targets_train.transpose(0,3,1,2)),\
                                   'mask_targets' : (('time','nc','lat','lon'),mask_targets_train.transpose(0,3,1,2)),\
                                   'NATL60'   : (('time','nc','lat','lon'),targets_mod[:,:,:,index[0]].transpose(0,3,1,2))},\
                        coords={'lon': indLon,\
                                'lat': indLat,\
                                'time': range(0,len(time_u))})
        data.to_netcdf(dirSAVE+"data_train.nc")

    return genFilename, meanTt, stdTt, inputs_train, mask_inputs_train, targets_train, mask_targets_train,\
           lday_train, inputs_train_OI, targets_mod, mask_targets_mod, targets_BFN


