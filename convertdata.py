import xarray as xr
import pickle
import numpy as np



def pickletoNetCDF(picklefile,savingpath,domain):
    with open(picklefile, 'rb') as handle:
        (gt_train, x_train_missing, x_train_pred, rec_AE_Tr, x_train_OI, \
         gt_test, x_test_missing, x_test_pred, rec_AE_Tt, x_test_OI) = pickle.load(handle)

    if domain == "OSMOSIS":
        extent = [-19.5, -11.5, 45., 55.]
        indLat = 200
        indLon = 160
    elif domain == 'GULFSTREAM':
        extent = [-65., -55., 33., 43.]
        indLat = 200
        indLon = 200
    else:
        extent = [-65., -55., 30., 40.]
        indLat = 200
        indLon = 200
    lon = np.arange(extent[0], extent[1], 1 / 20)
    lat = np.arange(extent[2], extent[3], 1 / 20)
    lon = lon[:indLon]
    lat = lat[:indLat]
    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    xrdata = xr.Dataset(\
        data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                   'latitude': (('lat', 'lon'), mesh_lat), \
                   'gt_train': (('time', 'lat', 'lon'), gt_train), \
                   'x_train_missing': (('time', 'lat', 'lon'), x_train_missing), \
                   'x_train_pred': (('time', 'lat', 'lon'), x_train_pred), \
                   'rec_AE_Tr': (('time', 'lat', 'lon'), rec_AE_Tr), \
                   'x_train_OI': (('time', 'lat', 'lon'), x_train_OI), \
                   'gt_test': (('time1', 'lat', 'lon'), gt_test), \
                   'x_test_missing': (('time1', 'lat', 'lon'), x_test_missing), \
                   'x_test_pred': (('time1', 'lat', 'lon'), x_test_pred), \
                   'rec_AE_Tt': (('time1', 'lat', 'lon'), rec_AE_Tt), \
                   'x_test_OI': (('time1', 'lat', 'lon'), x_test_OI)}, \
        coords={'lon': lon, 'lat': lat})
    xrdata.to_netcdf(path= savingpath, mode='w')
    return 1

if __name__ == '__main__':
    path="/home/q20febvr/research/4DVARNN-DinAE/out"
    file = path + "/saved_path_019_GENN_wwmissing.pickle"
    domain = "GULFSTREAM"
    savingpath = path + "/test.nc"
    pickletoNetCDF(file, savingpath, domain)
