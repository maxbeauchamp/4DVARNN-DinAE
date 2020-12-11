import os
from tools import *
from graphics import *


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


def plot_Figs(dict_global_Params,genFilename,genSuffixModel,\
              target_train,input_train,x_train_pred,rec_AE_Tr,x_train_OI,meanTr,stdTr,\
              target_test,input_test,x_test_pred,rec_AE_Tt,x_test_OI,\
              lday_pred,lday_test,iter):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    ## keep only the information on the target variable (remove covariates)
    if include_covariates == True:
        index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
        input_train     = input_train[:,index,:,:]
        input_test      = input_test[:,index,:,:]

    ## reshape and rescale variables
    input_train     = meanTr+ np.moveaxis(input_train,1,3)*stdTr
    target_train    = meanTr+ np.moveaxis(target_train,1,3)*stdTr
    x_train_pred    = meanTr+ np.moveaxis(x_train_pred,1,3)*stdTr
    rec_AE_Tr       = meanTr+ np.moveaxis(rec_AE_Tr,1,3)*stdTr
    input_test      = meanTr + np.moveaxis(input_test,1,3)*stdTr
    target_test     = meanTr+ np.moveaxis(target_test,1,3)*stdTr
    x_test_pred     = meanTr+ np.moveaxis(x_test_pred,1,3)*stdTr
    rec_AE_Tt       = meanTr+ np.moveaxis(rec_AE_Tt,1,3)*stdTr

    ## add OI (large-scale) to state if required
    if flagloadOIData == 1:

        input_train     = np.where(input_train==0.,np.nan,input_train+x_train_OI)
        target_train    = np.where(target_train==0.,np.nan,target_train+x_train_OI)
        x_train_pred    = x_train_pred + x_train_OI
        rec_AE_Tr       = rec_AE_Tr + x_train_OI
        input_test      = np.where(input_test==0.,np.nan,input_test+x_test_OI)
        target_test     = np.where(target_test==0.,np.nan,target_test+x_test_OI)
        x_test_pred     = x_test_pred + x_test_OI
        rec_AE_Tt       = rec_AE_Tt + x_test_OI

    ## generate some plots
    figpathTr = dirSAVE+'FIGS/Iter_%03d'%(iter)+'_Tr'
    if not os.path.exists(figpathTr):
        mk_dir_recursive(figpathTr)
    else:
        shutil.rmtree(figpathTr)
        mk_dir_recursive(figpathTr) 
    figpathTt = dirSAVE+'FIGS/Iter_%03d'%(iter)+'_Tt'
    if not os.path.exists(figpathTt):
        mk_dir_recursive(figpathTt)
    else:
        shutil.rmtree(figpathTt)
        mk_dir_recursive(figpathTt) 

    idT = int(np.floor(input_train.shape[3]/2))
    lon = np.arange(-65,-55,1/(20/dwscale))
    lat = np.arange(30,40,1/(20/dwscale))
    indLat     = np.arange(0,int(200/dwscale))
    indLon     = np.arange(0,int(200/dwscale))
    lon = lon[indLon]
    lat = lat[indLat]
    extent_=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    lfig=[20,40,60]

    ## Training dataset
    for ifig in lfig:
        # Rough variables
        figName = figpathTr+'/'+genFilename+genSuffixModel+'_examplesTr_%03d'%(ifig)+'_'+lday_pred[ifig]+'.png'
        fig, ax = plt.subplots(2,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.quantile(target_train[ifig,:,:,idT].flatten() , 0.05 )
        vmax = np.quantile(target_train[ifig,:,:,idT].flatten() , 0.95 )
        cmap="coolwarm"
        GT   = target_train[ifig,:,:,idT].squeeze()
        OBS  = input_train[ifig,:,:,idT].squeeze()
        PRED = x_train_pred[ifig,:,:,idT].squeeze()
        REC  = rec_AE_Tr[ifig,:,:,idT].squeeze()
        plot(ax,0,0,lon,lat,GT,"GT",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,0,1,lon,lat,OBS,"Observations",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,0,lon,lat,PRED,"Pred",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,1,lon,lat,REC,"Rec",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.subplots_adjust(hspace=0.5,wspace=0.25)
        plt.savefig(figName)       # save the figure
        plt.close()                # close the figure
        # Gradient
        figName = figpathTr+'/'+genFilename+genSuffixModel+'_examplesTr_grads_%03d'%(ifig)+'_'+lday_pred[ifig]+'.png'
        fig, ax = plt.subplots(2,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.quantile(Gradient(target_train[ifig,:,:,idT],2).flatten() , 0.05 )
        vmax = np.quantile(Gradient(target_train[ifig,:,:,idT],2).flatten() , 0.95 )
        cmap="viridis"
        GT   = Gradient(target_train[ifig,:,:,idT].squeeze(),2)
        OBS  = Gradient(input_train[ifig,:,:,idT].squeeze(),2)
        PRED = Gradient(x_train_pred[ifig,:,:,idT].squeeze(),2)
        REC  = Gradient(rec_AE_Tr[ifig,:,:,idT].squeeze(),2)
        plot(ax,0,0,lon,lat,GT,r"$\nabla_{GT}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,0,1,lon,lat,OBS,r"$\nabla_{Obs}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,0,lon,lat,PRED,r"$\nabla_{Pred}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,1,lon,lat,REC,r"$\nabla_{Rec}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.subplots_adjust(hspace=0.5,wspace=0.25)
        plt.savefig(figName)       # save the figure
        plt.close()                # close the figure

    ## Test dataset
    lfig=[5,10,15]
    for ifig in lfig:
        # Rough variables
        figName = figpathTt+'/'+genFilename+genSuffixModel+'_examplesTt_%03d'%(ifig)+'_'+lday_test[ifig]+'.png'
        fig, ax = plt.subplots(2,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.quantile(target_test[ifig,:,:,idT].flatten() , 0.05 )
        vmax = np.quantile(target_test[ifig,:,:,idT].flatten() , 0.95 )
        cmap="coolwarm"
        GT   = target_test[ifig,:,:,idT].squeeze()
        OBS  = input_test[ifig,:,:,idT].squeeze()
        PRED = x_test_pred[ifig,:,:,idT].squeeze()
        REC  = rec_AE_Tt[ifig,:,:,idT].squeeze()
        plot(ax,0,0,lon,lat,GT,"GT",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,0,1,lon,lat,OBS,"Observations",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,0,lon,lat,PRED,"Pred",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,1,lon,lat,REC,"Rec",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.subplots_adjust(hspace=0.5,wspace=0.25)
        plt.savefig(figName)       # save the figure
        plt.close()                # close the figure
        # Gradient variables
        figName = figpathTt+'/'+genFilename+genSuffixModel+'_examplesTt_grads_%03d'%(ifig)+'_'+lday_test[ifig]+'.png'
        fig, ax = plt.subplots(2,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.quantile(Gradient(target_test[ifig,:,:,idT],2).flatten() , 0.05 )
        vmax = np.quantile(Gradient(target_test[ifig,:,:,idT],2).flatten() , 0.95 )
        cmap="viridis"
        GT   = Gradient(target_test[ifig,:,:,idT].squeeze(),2)
        OBS  = Gradient(input_test[ifig,:,:,idT].squeeze(),2)
        PRED = Gradient(x_test_pred[ifig,:,:,idT].squeeze(),2)
        REC  = Gradient(rec_AE_Tt[ifig,:,:,idT].squeeze(),2)
        plot(ax,0,0,lon,lat,GT,r"$\nabla_{GT}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,0,1,lon,lat,OBS,r"$\nabla_{Observations}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,0,lon,lat,PRED,r"$\nabla_{Pred}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,1,lon,lat,REC,r"$\nabla_{Rec}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.subplots_adjust(hspace=0.5,wspace=0.25)
        plt.savefig(figName)       # save the figure
        plt.close()                # close the figure


def plot_Figs2(dict_global_Params,genFilename,genSuffixModel,\
              x_target,mask_target,x_input,mask_input,\
              x_train_pred,rec_AE_Tr,x_train_OI,meanTr,stdTr,\
              lday_pred,iter):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    # import BFN results
    BFN_10=xr.open_dataset("/users/local/DATA/OSE/GULFSTREAM/OSE_GULFSTREAM_BFN_daily.nc")
    ## Apply reduction parameter
    if dwscale!=1:
        BFN_10 = einops.reduce(BFN_10.SSH.values,  '(t t1) (h h1) (w w1) -> t h w', t1=1, h1=int(dwscale/2), w1=int(dwscale/2), reduction=np.nanmean)

    ## keep only the information on the target variable (remove covariates)
    if include_covariates == True:
        index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
        mask_input  = mask_input[:,index,:,:,]
        x_input     = x_input[:,index,:,:,]

    ## reshape and rescale variables
    mask_input       = np.moveaxis(mask_input,1,3)
    x_input          = meanTr+ np.moveaxis(x_input,1,3)*stdTr
    mask_target      = np.moveaxis(mask_target,1,3)
    x_target         = meanTr+ np.moveaxis(x_target,1,3)*stdTr
    x_train_pred     = meanTr+ np.moveaxis(x_train_pred,1,3)*stdTr
    rec_AE_Tr        = meanTr+ np.moveaxis(rec_AE_Tr,1,3)*stdTr

    ## add OI (large-scale) to state if required
    if flagloadOIData == 1:
        x_input         = x_input + x_train_OI
        x_target        = x_target + x_train_OI
        x_train_pred    = x_train_pred + x_train_OI
        rec_AE_Tr       = rec_AE_Tr + x_train_OI

    ## generate some plots
    figpathTr = dirSAVE+'FIGS/Iter_%03d'%(iter)+'_Tr'
    if not os.path.exists(figpathTr):
        mk_dir_recursive(figpathTr)
    else:
        shutil.rmtree(figpathTr)
        mk_dir_recursive(figpathTr)

    idT = int(np.floor(x_input.shape[3]/2))
    lon = np.arange(-65,-55,1/(20/dwscale))
    lat = np.arange(30,40,1/(20/dwscale))
    indLat     = np.arange(0,int(200/dwscale))
    indLon     = np.arange(0,int(200/dwscale))
    lon = lon[indLon]
    lat = lat[indLat]
    extent_=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    lfig=[20,40,60]

    ## Training dataset
    for ifig in lfig:
        # Rough variables
        figName = figpathTr+'/'+genFilename+genSuffixModel+'_examplesTr_%03d'%(ifig)+'_'+lday_pred[ifig]+'.png'
        fig, ax = plt.subplots(3,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.nanquantile(x_train_pred[ifig,:,:,idT].flatten() , 0.05 )
        vmax = np.nanquantile(x_train_pred[ifig,:,:,idT].flatten() , 0.95 )
        cmap="coolwarm"
        GT   = np.where(mask_target[ifig,:,:,idT].squeeze()==0,
                 np.nan,x_target[ifig,:,:,idT].squeeze())
        OBS  = np.where(mask_input[ifig,:,:,idT].squeeze()==0,\
                 np.nan, x_input[ifig,:,:,idT].squeeze())
        OI   = x_train_OI[ifig,:,:,idT].squeeze() 
        PRED = x_train_pred[ifig,:,:,idT].squeeze()
        REC  = rec_AE_Tr[ifig,:,:,idT].squeeze()
        BFN  = BFN_10[ifig]
        plot(ax,0,0,lon,lat,GT,"GT",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,0,1,lon,lat,OBS,"Observations",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,0,lon,lat,PRED,"Pred",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        #plot(ax,1,1,lon,lat,REC,"Rec",\
        #     extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,1,lon,lat,BFN,"BFN",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,2,0,lon,lat,OI,"OI",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,2,1,lon,lat,OI-PRED,"Anomaly (OI-GENN)",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.subplots_adjust(hspace=0.5,wspace=0.25)
        plt.savefig(figName)       # save the figure
        plt.close()                # close the figure
        # Gradient
        figName = figpathTr+'/'+genFilename+genSuffixModel+'_examplesTr_grads_%03d'%(ifig)+'_'+lday_pred[ifig]+'.png'
        fig, ax = plt.subplots(3,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.nanquantile(Gradient(x_train_pred[ifig,:,:,idT],2).flatten() , 0.05 )
        vmax = np.nanquantile(Gradient(x_train_pred[ifig,:,:,idT],2).flatten() , 0.95 )
        cmap="viridis"
        GT   = Gradient(np.where(mask_target[ifig,:,:,idT].squeeze()==0,
                 np.nan,x_target[ifig,:,:,idT].squeeze()),2)
        OBS  = Gradient(np.where(mask_input[ifig,:,:,idT].squeeze()==0,\
                 np.nan, x_input[ifig,:,:,idT].squeeze()),2)
        OI   = Gradient(x_train_OI[ifig,:,:,idT].squeeze(),2)
        PRED = Gradient(x_train_pred[ifig,:,:,idT].squeeze(),2)
        REC  = Gradient(rec_AE_Tr[ifig,:,:,idT].squeeze(),2)
        BFN  = Gradient(BFN,2)
        ANOM = Gradient(x_train_OI[ifig,:,:,idT].squeeze()-x_train_pred[ifig,:,:,idT].squeeze(),2)
        plot(ax,0,0,lon,lat,GT,r"$\nabla_{GT}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,0,1,lon,lat,OBS,r"$\nabla_{Obs}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,0,lon,lat,PRED,r"$\nabla_{Pred}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        #plot(ax,1,1,lon,lat,REC,r"$\nabla_{Rec}$",\
        #     extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,1,1,lon,lat,BFN,r"$\nabla_{BFN}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,2,0,lon,lat,OI,r"$\nabla_{OI}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plot(ax,2,1,lon,lat,ANOM,r"$\nabla_{Anomaly}$",\
             extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.subplots_adjust(hspace=0.5,wspace=0.25)
        plt.savefig(figName)       # save the figure
        plt.close()                # close the figure



