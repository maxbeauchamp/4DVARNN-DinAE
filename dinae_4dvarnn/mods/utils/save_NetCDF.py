import os
from tools import *
from graphics import *

def save_NetCDF(dict_global_Params,\
                meanTr,stdTr,\
                x_test,x_test_missing,x_test_pred,rec_AE_Tt,x_test_OI,\
                iter):     

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    if domain=="OSMOSIS":
        extent     = [-19.5,-11.5,45.,55.]
        indLat     = 200
        indLon     = 160
    elif domain=='GULFSTREAM':
        extent     = [-65.,-55.,33.,43.]
        indLat     = 200
        indLon     = 200
    else:
        extent=[-65.,-55.,30.,40.]
        indLat     = 200
        indLon     = 200

    lon = np.arange(extent[0],extent[1],1/(20/dwscale))
    lat = np.arange(extent[2],extent[3],1/(20/dwscale))
    indLat     = int(indLat/dwscale)
    indLon     = int(indLon/dwscale)
    lon = lon[:indLon]
    lat = lat[:indLat]
    extent_=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    saved_path1 = dirSAVE+'/NATL60_GULFSTREAM_XP'+ixp+'_'+suf3+"_"+suf1+"_%03d.nc"%(iter)
    saved_path2 = dirSAVE+'/NATL60_GULFSTREAM_XP'+ixp+'_rec_'+suf3+"_"+suf1+"_%03d.nc"%(iter)

    ## keep only the information on the target variable (remove covariates)
    if include_covariates == True:
        index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
        x_test_missing  = x_test_missing[:,index,:,:]

    ## reshape and rescale variables
    x_test          = meanTr+ np.moveaxis(x_test,1,3)*stdTr
    x_test_missing  = meanTr+ np.moveaxis(x_test_missing,1,3)*stdTr
    x_test_pred     = meanTr+ np.moveaxis(x_test_pred,1,3)*stdTr
    rec_AE_Tt       = meanTr+ np.moveaxis(rec_AE_Tt,1,3)*stdTr

    ## add OI (large-scale) to state if required
    if flagloadOIData == 1:
        x_test          = x_test + x_test_OI
        x_test_missing  = x_test_missing + x_test_OI
        x_test_pred     = x_test_pred + x_test_OI
        rec_AE_Tt       = rec_AE_Tt + x_test_OI

    idT = int(np.floor(x_test.shape[3]/2))
    indN_Tt = np.concatenate([np.arange(60,80),np.arange(140,160),\
                             np.arange(220,240),np.arange(300,320)])
    time    = [ datetime.strftime(datetime.strptime("2012-10-01",'%Y-%m-%d')\
                          + timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tt ]
    # interpolation
    xrdata = xr.Dataset(\
                data_vars={'longitude': (('lat','lon'),mesh_lon),\
                           'latitude' : (('lat','lon'),mesh_lat),\
                           'Time'     : (('time'),time),\
                           'ssh'  : (('time','lat','lon'),x_test_pred[:,:,:,idT])},\
                coords={'lon': lon,'lat': lat,'time': indN_Tt})
    xrdata.time.attrs['units']='days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=saved_path1, mode='w')
    # reconstruction
    xrdata = xr.Dataset(\
                data_vars={'longitude': (('lat','lon'),mesh_lon),\
                           'latitude' : (('lat','lon'),mesh_lat),\
                           'Time'     : (('time'),time),\
                           'ssh'  : (('time','lat','lon'),rec_AE_Tt[:,:,:,idT])},\
                coords={'lon': lon,'lat': lat,'time': indN_Tt})
    xrdata.time.attrs['units']='days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=saved_path2, mode='w')
