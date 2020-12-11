#%%
import Interpolationeval
import numpy as np
#%%
#pickle file and savingfile defintions
"""
path="/users/local/m20amar/gpfsscratch/rech/yrf/uba22to/4DVARNN-DINAE/domain=GULFSTREAM/resIA_nadirswot_nadlag_lag=5_type_obs=obs/GB1_GENN_wmissing_wOI"
file = path + "/saved_path_007_GENN_wmissing.pickle"
"""
#%%
domain = "GULFSTREAM"
file1 = '/home/q20febvr/research/4DVARNN-DinAE/out/test.nc'
savingpath = file1
AEflag = True
#convert the pickle file to NetCDF
#convertdata.pickletoNetCDF(file, savingpath, domain)
#evaluate the interpolation results
nRMSE,R,I,nRMSE_OI,R_OI,I_OI,AE, nRMSE_Grad,R_Grad,I_Grad,nRMSE_Grad_OI,R_Grad_OI,I_Grad_OI,AE_Grad=Interpolationeval.Scoreslist_test(savingpath,domain,True)
#%%
#plot the figures
Interpolationeval.PlotFigures(nRMSE,R,I,nRMSE_OI,R_OI,I_OI,AE, nRMSE_Grad,R_Grad,I_Grad,nRMSE_Grad_OI,R_Grad_OI,I_Grad_OI,AE_Grad)
print("nRMSE = ", np.mean(nRMSE))
print("R = ", np.mean(R))
print("I = ", np.mean(I))
print("nRMSE_OI = ", np.mean(nRMSE_OI))
print("R_OI = ", np.mean(R_OI))
print("I_OI = ", np.mean(I_OI))
print("AE = ", np.mean(AE))
print("Gradient evaluation")
print("nRMSE_Grad = ", np.mean(nRMSE_Grad))
print("R_Grad= ", np.mean(R_Grad))
print("I_Grad = ", np.mean(I_Grad))
print("nRMSE_Grad_OI = ", np.mean(nRMSE_Grad_OI))
print("R_Grad_OI = ", np.mean(R_Grad_OI))
print("I_Grad_OI = ", np.mean(I_Grad_OI))
print("AE_Grad = ", np.mean(AE_Grad))


Interpolationeval.plots(savingpath,0)
#animate.Animate_SSH(savingpath)