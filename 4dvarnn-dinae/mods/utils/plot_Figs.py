import os
from .tools import *
from .graphics import *

def plot_Figs(dirSAVE,genFilename,genSuffixModel,\
              x_train,x_train_missing,mask_train,x_train_pred,rec_AE_Tr,\
              x_test,x_test_missing,mask_test,lday_test,x_test_pred,rec_AE_Tt,\
              iter):
 
    # generate some plots
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

    idT = int(np.floor(x_test.shape[3]/2))
    lon = np.arange(-65,-55,1/20)
    lat = np.arange(30,40,1/20)
    indLat     = np.arange(0,200)
    indLon     = np.arange(0,200)
    lon = lon[indLon]
    lat = lat[indLat]
    extent_=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    lfig=[20,40,60]

    # Training dataset
    for ifig in lfig:

        # Rough variables
        figName = figpathTr+'/'+genFilename+genSuffixModel+'_examplesTr_%03d'%(ifig)+'.png' 
        fig, ax = plt.subplots(2,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.quantile(x_train[ifig,:,:,idT].flatten() , 0.05 )
        vmax = np.quantile(x_train[ifig,:,:,idT].flatten() , 0.95 )
        cmap="coolwarm"
        GT   = x_train[ifig,:,:,idT].squeeze()
        OBS  = np.where(mask_train[ifig,:,:,idT].squeeze()==0,\
                 np.nan, x_train_missing[ifig,:,:,idT].squeeze())
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
        figName = figpathTr+'/'+genFilename+genSuffixModel+'_examplesTr_grads_%03d'%(ifig)+'.png'
        fig, ax = plt.subplots(2,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.quantile(Gradient(x_train[ifig,:,:,idT],2).flatten() , 0.05 )
        vmax = np.quantile(Gradient(x_train[ifig,:,:,idT],2).flatten() , 0.95 )
        cmap="viridis"
        GT   = Gradient(x_train[ifig,:,:,idT].squeeze(),2)
        OBS  = Gradient(np.where(mask_train[ifig,:,:,idT].squeeze()==0,\
                 np.nan,x_train_missing[ifig,:,:,idT].squeeze()),2)
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

    # Test dataset
    lfig=[15,30,45]
    for ifig in lfig:

        # Rough variables
        figName = figpathTt+'/'+genFilename+genSuffixModel+'_examplesTt_%03d'%(ifig)+'_'+lday_test[ifig]+'.png'
        fig, ax = plt.subplots(2,2,figsize=(15,15),
                      subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
        vmin = np.quantile(x_test[ifig,:,:,idT].flatten() , 0.05 )
        vmax = np.quantile(x_test[ifig,:,:,idT].flatten() , 0.95 )
        cmap="coolwarm"
        GT   = x_test[ifig,:,:,idT].squeeze()
        OBS  = np.where(mask_test[ifig,:,:,idT].squeeze()==0,\
                 np.nan, x_test_missing[ifig,:,:,idT].squeeze())
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
        vmin = np.quantile(Gradient(x_test[ifig,:,:,idT],2).flatten() , 0.05 )
        vmax = np.quantile(Gradient(x_test[ifig,:,:,idT],2).flatten() , 0.95 )
        cmap="viridis"
        GT   = Gradient(x_test[ifig,:,:,idT].squeeze(),2)
        OBS  = Gradient(np.where(mask_test[ifig,:,:,idT].squeeze()==0,\
                 np.nan, x_test_missing[ifig,:,:,idT].squeeze()),2)
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

