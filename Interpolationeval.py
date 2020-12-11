# %%
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
#%%
def gradient_norm(a):
    return np.array(np.linalg.norm(np.gradient(a),axis=0))
#%%
def Scoreslist_train(file1,domain,AEflag):
    #%%

    #%%
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
    #%%
    #Training data#
    nc_data_train = Dataset(file1, "r")
    gt_train = np.array(nc_data_train['gt_train'])
    x_train_missing = np.array(nc_data_train['x_train_missing'])

    x_train_pred = np.array(nc_data_train['x_train_pred'])
    rec_AE_Tr = np.array(nc_data_train['rec_AE_Tr'])
    x_train_OI = np.array(nc_data_train['x_train_OI'])
    mask_train = np.array(nc_data_train['gt_train']) == np.array(nc_data_train['x_train_OI'])
    #%%

    ##*** INIT SSH STATISTICS ***##
    ## Init variables for temporal analysis (nRMSE scores)
    nRMSE_Tr= np.zeros(len(gt_train))
    ## Init variables for temporal analysis (R scores)
    R_Tr = np.zeros(len(gt_train))
    ## Init variables for temporal analysis (I scores)
    I_Tr= np.zeros(len(gt_train))
    ## Init variables for temporal analysis (nRMSE scores)
    nRMSE_OI_Tr = np.zeros(len(gt_train))
    ## Init variables for temporal analysis (R scores)
    R_OI_Tr = np.zeros(len(gt_train))
    ## Init variables for temporal analysis (I scores)
    I_OI_Tr = np.zeros(len(gt_train))
    if AEflag == True:
        ## Init variables for temporal analysis (AE scores)
        AE_Tr = np.zeros(len(gt_train))

    ##*** INIT GradSSH STATISTICS ***##
    ## Init variables for temporal analysis (nRMSE scores)
    nrmse_Grad_Tr = np.zeros(len(gt_train))
    ## Init variables for temporal analysis (R scores)
    R_Grad_Tr = np.zeros(len(gt_train))
    ## Init variables for temporal analysis (I scores)
    I_Grad_Tr= np.zeros(len(gt_train))
    ## Init variables for temporal analysis (nRMSE scores)
    nrmse_Grad_OI_Tr = np.zeros(len(gt_train))
    ## Init variables for temporal analysis (R scores)
    R_Grad_OI_Tr = np.zeros(len(gt_train))
    ## Init variables for temporal analysis (I scores)
    I_Grad_OI_Tr = np.zeros(len(gt_train))
    if AEflag == True:
        ## Init variables for temporal analysis (AE scores)
        AE_Grad_tr = np.zeros(len(gt_train))

    mask_train2 = mask_train == 0
    print(mask_train2.shape)
    mask_train2 = np.array(mask_train2).astype(int)
    print(mask_train2[0])


    for i in range(0,len(gt_train)):
        ## Compute NRMSE statistics (i.e. RMSE/stdev(gt))
        nRMSE_Tr[i] = (np.sqrt(np.nanmean(((gt_train[i]-np.nanmean(gt_train[i]))-(x_train_pred[i]-np.nanmean(x_train_pred[i])))**2)))/np.nanstd(gt_train[i])
        ## Compute R scores
        R_Tr[i] = 100 * (1 - np.nanmean(((mask_train[i] * gt_train[i] - np.nanmean(mask_train[i] * gt_train[i])) - (mask_train[i] * x_train_pred[i] - np.nanmean(mask_train[i] * x_train_pred[i]))) ** 2) / np.nanvar(mask_train[i] * gt_train[i]))
        ## Compute I scores
        I_Tr[i] = 100 * (1 - np.nanmean(((mask_train2[i] * gt_train[i] - np.nanmean(mask_train2[i] * gt_train[i])) - (mask_train2[i] * x_train_pred[i] - np.nanmean(mask_train2[i] * x_train_pred[i]))) ** 2) / np.nanvar(mask_train2[i] * gt_train[i]))

        nRMSE_OI_Tr[i] = (np.sqrt(np.nanmean(((gt_train[i] - np.nanmean(gt_train[i])) - (x_train_OI[i] - np.nanmean(x_train_OI[i]))) ** 2))) / np.nanstd(gt_train[i])
        ## Compute R scores
        R_OI_Tr[i] = 100 * (1 - np.nanmean(((mask_train[i] * gt_train[i] - np.nanmean(mask_train[i] * gt_train[i])) - (mask_train[i] * x_train_OI[i] - np.nanmean(mask_train[i] * x_train_OI[i]))) ** 2) / np.nanvar(mask_train[i] * gt_train[i]))
        ## Compute I scores
        I_OI_Tr[i] = 100 * (1 - np.nanmean(((mask_train2[i] * gt_train[i] - np.nanmean(mask_train2[i] * gt_train[i])) - (mask_train2[i] * x_train_OI[i] - np.nanmean(mask_train2[i] * x_train_OI[i]))) ** 2) / np.nanvar(mask_train2[i] * gt_train[i]))
        if AEflag == True:
            ## Compute AE scores
            AE_Tr[i] = 100 * (1 - np.nanmean(((gt_train[i] - np.nanmean(gt_train[i])) - (rec_AE_Tr[i] - np.nanmean(rec_AE_Tr[i]))) ** 2) / np.nanvar(gt_train[i]))

    return (nRMSE_Tr,R_Tr,I_Tr,nRMSE_OI_Tr,R_OI_Tr,I_OI_Tr,AE_Tr)

def Scoreslist_test(file1,domain,AEflag):
    #%%
    domain = "GULFSTREAM"
    file1 = '/home/q20febvr/research/4DVARNN-DinAE/out/test.nc'
    savingpath = file1
    AEflag = True
    #%%
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
    #Test data#
    nc_data_test = Dataset(file1, "r")
    gt_test = np.array(nc_data_test['gt_test'])
    x_test_missing = np.array(nc_data_test['x_test_missing'])
    #%%
    x_test_pred = np.array(nc_data_test['x_test_pred'])
    rec_AE_Tt = np.array(nc_data_test['rec_AE_Tt'])
    x_test_OI = np.array(nc_data_test['x_test_OI'])
    mask_test = np.array(nc_data_test['gt_test']) == np.array(nc_data_test['x_test_OI'])
    gt_grad_test = gradient_norm(np.array(nc_data_test['gt_test']))
    #%%

    x_grad_test_missing = gradient_norm(np.array(nc_data_test['x_test_missing']))
    x_grad_test_pred = gradient_norm(np.array(nc_data_test['x_test_pred']))
    rec_AE_grad_Tt = gradient_norm(np.array(nc_data_test['rec_AE_Tt']))
    x_grad_test_OI = gradient_norm(np.array(nc_data_test['x_test_OI']))

    ##*** INIT SSH STATISTICS ***##
    ## Init variables for temporal analysis (nRMSE scores)
    nRMSE_Tt= np.zeros(len(gt_test))
    ## Init variables for temporal analysis (R scores)
    R_Tt = np.zeros(len(gt_test))
    ## Init variables for temporal analysis (I scores)
    I_Tt= np.zeros(len(gt_test))
    ## Init variables for temporal analysis (nRMSE scores)
    nRMSE_OI_Tt = np.zeros(len(gt_test))
    ## Init variables for temporal analysis (R scores)
    R_OI_Tt = np.zeros(len(gt_test))
    ## Init variables for temporal analysis (I scores)
    I_OI_Tt = np.zeros(len(gt_test))
    if AEflag == True:
        ## Init variables for temporal analysis (AE scores)
        AE_Tt = np.zeros(len(gt_test))

    ##*** INIT GradSSH STATISTICS ***##
    ## Init variables for temporal analysis (nRMSE scores)
    nRMSE_Grad_Tt = np.zeros(len(gt_test))
    ## Init variables for temporal analysis (R scores)
    R_Grad_Tt = np.zeros(len(gt_test))
    ## Init variables for temporal analysis (I scores)
    I_Grad_Tt= np.zeros(len(gt_test))
    ## Init variables for temporal analysis (nRMSE scores)
    nRMSE_Grad_OI_Tt = np.zeros(len(gt_test))
    ## Init variables for temporal analysis (R scores)
    R_Grad_OI_Tt = np.zeros(len(gt_test))
    ## Init variables for temporal analysis (I scores)
    I_Grad_OI_Tt = np.zeros(len(gt_test))
    if AEflag == True:
        ## Init variables for temporal analysis (AE scores)
        AE_Grad_Tt = np.zeros(len(gt_test))

    mask_test2 = (mask_test == 0)
    print(mask_test2.shape)
    mask_test2 = np.array(mask_test2).astype(int)

    for i in range(0,len(gt_test)):
        ## Compute NRMSE statistics (i.e. RMSE/stdev(gt))
        nRMSE_Tt[i] = (np.sqrt(np.nanmean(((gt_test[i]-np.nanmean(gt_test[i]))-(x_test_pred[i]-np.nanmean(x_test_pred[i])))**2)))/np.nanstd(gt_test[i])
        ## Compute R scores
        R_Tt[i] = 100 * (1 - np.nanmean(((mask_test[i] * gt_test[i] - np.nanmean(mask_test[i] * gt_test[i])) - (mask_test[i] * x_test_pred[i] - np.nanmean(mask_test[i] * x_test_pred[i]))) ** 2) / np.nanvar(mask_test[i] * gt_test[i]))
        ## Compute I scores
        I_Tt[i] = 100 * (1 - np.nanmean(((mask_test2[i] * gt_test[i] - np.nanmean(mask_test2[i] * gt_test[i])) - (mask_test2[i] * x_test_pred[i] - np.nanmean(mask_test2[i] * x_test_pred[i]))) ** 2) / np.nanvar(mask_test2[i] * gt_test[i]))

        nRMSE_OI_Tt[i] = (np.sqrt(np.nanmean(((gt_test[i] - np.nanmean(gt_test[i])) - (x_test_OI[i] - np.nanmean(x_test_OI[i]))) ** 2))) / np.nanstd(gt_test[i])
        ## Compute R scores
        R_OI_Tt[i] = 100 * (1 - np.nanmean(((mask_test[i] * gt_test[i] - np.nanmean(mask_test[i] * gt_test[i])) - (mask_test[i] * x_test_OI[i] - np.nanmean(mask_test[i] * x_test_OI[i]))) ** 2) / np.nanvar(mask_test[i] * gt_test[i]))
        ## Compute I scores
        I_OI_Tt[i] = 100 * (1 - np.nanmean(((mask_test2[i] * gt_test[i] - np.nanmean(mask_test2[i] * gt_test[i])) - (mask_test2[i] * x_test_OI[i] - np.nanmean(mask_test2[i] * x_test_OI[i]))) ** 2) / np.nanvar(mask_test2[i] * gt_test[i]))
        if AEflag == True:
            ## Compute AE scores
            AE_Tt[i] = 100 * (1 - np.nanmean(((gt_test[i] - np.nanmean(gt_test[i])) - (rec_AE_Tt[i] - np.nanmean(rec_AE_Tt[i]))) ** 2) / np.nanvar(gt_test[i]))

        ###Gradient####
        ## Compute NRMSE statistics (i.e. RMSE/stdev(gt))
        nRMSE_Grad_Tt[i] = (np.sqrt(np.nanmean(((gt_grad_test[i] - np.nanmean(gt_grad_test[i])) - (x_grad_test_pred[i] - np.nanmean(x_grad_test_pred[i]))) ** 2))) / np.nanstd(gt_grad_test[i])
        ## Compute R scores
        R_Grad_Tt[i] = 100 * (1 - np.nanmean(((mask_test[i] * gt_grad_test[i] - np.nanmean(mask_test[i] * gt_grad_test[i])) - (mask_test[i] * x_grad_test_pred[i] - np.nanmean(mask_test[i] * x_grad_test_pred[i]))) ** 2) / np.nanvar(mask_test[i] * gt_grad_test[i]))
        ## Compute I scores
        I_Grad_Tt[i] = 100 * (1 - np.nanmean(((mask_test2[i] * gt_grad_test[i] - np.nanmean(mask_test2[i] * gt_grad_test[i])) - (mask_test2[i] * x_grad_test_pred[i] - np.nanmean(mask_test2[i] * x_grad_test_pred[i]))) ** 2) / np.nanvar(mask_test2[i] * gt_grad_test[i]))

        nRMSE_Grad_OI_Tt[i] = (np.sqrt(np.nanmean(((gt_grad_test[i] - np.nanmean(gt_grad_test[i])) - (x_grad_test_OI[i] - np.nanmean(x_grad_test_OI[i]))) ** 2))) / np.nanstd(gt_grad_test[i])
        ## Compute R scores
        R_Grad_OI_Tt[i] = 100 * (1 - np.nanmean(((mask_test[i] * gt_grad_test[i] - np.nanmean(mask_test[i] * gt_grad_test[i])) - (mask_test[i] * x_grad_test_OI[i] - np.nanmean(mask_test[i] * x_grad_test_OI[i]))) ** 2) / np.nanvar(mask_test[i] * gt_grad_test[i]))
        ## Compute I scores
        I_Grad_OI_Tt[i] = 100 * (1 - np.nanmean(((mask_test2[i] * gt_grad_test[i] - np.nanmean(mask_test2[i] * gt_grad_test[i])) - (mask_test2[i] * x_grad_test_OI[i] - np.nanmean(mask_test2[i] * x_grad_test_OI[i]))) ** 2) / np.nanvar(mask_test2[i] * gt_grad_test[i]))
        if AEflag == True:
            ## Compute AE scores
            AE_Grad_Tt[i] = 100 * (1 - np.nanmean(((gt_grad_test[i] - np.nanmean(gt_grad_test[i])) - (rec_AE_grad_Tt[i] - np.nanmean(rec_AE_grad_Tt[i]))) ** 2) / np.nanvar(gt_grad_test[i]))

    return (nRMSE_Tt,R_Tt,I_Tt,nRMSE_OI_Tt,R_OI_Tt,I_OI_Tt,AE_Tt,nRMSE_Grad_Tt, R_Grad_Tt, I_Grad_Tt, nRMSE_Grad_OI_Tt, R_Grad_OI_Tt, I_Grad_OI_Tt, AE_Grad_Tt)



def PlotFigures(nRMSE,R,I,nRMSE_OI,R_OI,I_OI,AE, nRMSE_Grad,R_Grad,I_Grad,nRMSE_Grad_OI,R_Grad_OI,I_Grad_OI,AE_Grad):
    days = np.arange(len(nRMSE))
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(days,nRMSE)
    plt.plot(days, nRMSE_OI)
    plt.legend(["nRMSE", "nRMSE_OI"])
    plt.title("nRMSEs")
    plt.subplot(1,2,2)
    plt.plot(days,R)
    plt.plot(days,I)
    plt.plot(days,R_OI)
    plt.plot(days,I_OI)
    plt.plot(days,AE)
    plt.legend(["R","I","R_OI","I_OI","AE"])
    plt.title("Reconstruction Scores")
    #plt.savefig("/homes/m20amar/4DVAR/MyFigs/model2_test.png")
    plt.show()

    #gradient plots
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.plot(days, nRMSE_Grad)
    plt.plot(days, nRMSE_Grad_OI)
    plt.legend(["nRMSE_Grad", "nRMSE_Grad_OI"])
    plt.title("nRMSEs_Grad")
    plt.subplot(1, 2, 2)
    plt.plot(days, R_Grad)
    plt.plot(days, I_Grad)
    plt.plot(days, R_Grad_OI)
    plt.plot(days, I_Grad_OI)
    plt.plot(days, AE_Grad)
    plt.legend(["R_Grad", "I_Grad", "R_Grad_OI", "I_Grad_OI", "AE_Grad"])
    plt.title("Reconstruction Scores")
    # plt.savefig("/homes/m20amar/4DVAR/MyFigs/model2_test.png")
    plt.show()
    return 1

def plots(file1,ind):
    nc_data_test = Dataset(file1, "r")
    gt_test = np.array(nc_data_test['gt_test'])
    x_test_missing = np.array(nc_data_test['x_test_missing'])
    mask_test = np.array(nc_data_test['gt_test']) == np.array(nc_data_test['x_test_OI'])
    x_test_pred = np.array(nc_data_test['x_test_pred'])
    rec_AE_Tt = np.array(nc_data_test['rec_AE_Tt'])
    x_test_OI = np.array(nc_data_test['x_test_OI'])

    gt_grad_test = gradient_norm(np.array(nc_data_test['gt_test']))
    x_grad_test_missing = gradient_norm(np.array(nc_data_test['x_test_missing']))
    x_grad_test_pred = gradient_norm(np.array(nc_data_test['x_test_pred']))
    rec_AE_grad_Tt = gradient_norm(np.array(nc_data_test['rec_AE_Tt']))
    x_grad_test_OI = gradient_norm(np.array(nc_data_test['x_test_OI']))

    plt.figure(1)

    plt.subplot(2,2,1)
    plt.imshow(gt_test[ind])
    plt.title("GT")
    plt.subplot(2, 2, 2)
    #plt.imshow(mask_test[ind]*gt_test[ind])
    plt.imshow(x_test_missing[ind])
    plt.title("Obs")
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(x_test_pred[ind])
    plt.title("Pred")
    plt.subplot(2, 2, 4)
    plt.imshow(x_test_OI[ind])
    plt.title("OI")
    plt.show()

    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.imshow(gt_grad_test[ind])
    plt.title(r"$\nabla_{GT}$")
    plt.subplot(2, 2, 2)
    #plt.imshow(mask_test[ind] * gt_grad_test[ind])
    plt.imshow(x_grad_test_missing[ind])
    plt.title(r"$\nabla_{Obs}$")
    plt.subplot(2, 2, 3)
    plt.imshow(x_grad_test_pred[ind])
    plt.title(r"$\nabla_{Pred}$")
    plt.subplot(2, 2, 4)
    plt.imshow(x_grad_test_OI[ind])
    plt.title(r"$\nabla_{OI}$")
    plt.show()

    return 1
























