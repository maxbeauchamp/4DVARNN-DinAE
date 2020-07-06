#import tensorflow.keras as keras
#import keras
import time
import copy
import torch
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F
import dinAE_solver_torch as dinAE

# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #parser.add_argument('-d', '--data', help='Image dataset used for learning: cifar, mnist or custom numpy 4D array (samples, rows, colums, channel) using pickle', type=str, default='cifar')
    #parser.add_argument('-e', '--epoch', help='Number of epochs', type=int, default=100)
    #parser.add_argument('-b', '--batchsize', help='Batch size', type=int, default=64)
    #parser.add_argument('-o', '--output', help='Output model name (required) (.h5)', type=str, required = True)
    #parser.add_argument('--optim', help='Optimization method: sgd, adam', type=str, default='sgd')

    flagDisplay   = 0
    flagProcess   = [0,1,2,3,4]
    flagSaveModel = 1
     
    Wsquare     = 0#0 # half-width of holes
    #Nsquare     = 3  # number of holes
    DimAE       = 20#20 # Dimension of the latent space
    flagAEType  = 7#7#

    flagDataset       = 2 # 0: MNIST, 1: MNIST-FASHION, 2: SST
    flagDataWindowing = 0
    flagloadOIData    = 0
    
    
    alpha_Losss     = np.array([1.,0.1])
    flagGradModel   = 0 # Gradient computation (0: subgradient, 1: true gradient/autograd)
    flagOptimMethod = 1 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
    #NiterProjection = 5 # Number of fixed-point iterations
    #NiterGrad       = 5 # Number of gradient descent step
  
    batch_size        = 4#4#8#12#8#256#8
    NbEpoc            = 20
    Niter             = 50
    NSampleTr         = 445#550#334#
    
    for kk in range(0,len(flagProcess)):

        ## import data
        if flagProcess[kk] == 0:  
            cf import_Datasets.py

        ## define AE architecture
        elif flagProcess[kk] == 2:                    
            DimCAE = DimAE
            shapeData     = np.array(x_train.shape[1:])          ******
            shapeData[0]  = x_train.shape[3]                     ******
            shapeData[1:] = x_train.shape[1:3]                   ******
                
            elif flagAEType == 6: ## Conv-AE for SST
              cf flagProces_2.6.py
            elif flagAEType == 7: ## GENN
               cf flagProces_2.7.py
            elif flagAEType == 8: ## two-scale GENN
               cf flagProces_2.8.py
              
            model_AE = Model_AE()    ******
            print(model_AE)          ******
            print('Number of trainable parameters = %d'%(sum(p.numel() for p in model_AE.parameters() if p.requires_grad)))  ******

        ## train Conv-AE
        elif flagProcess[kk] == 4:        

        ###############################################################
        ## train Conv-AE using test data with missing data
        elif flagProcess[kk] == 5:        

        ###############################################################
        ## Performance statistics Conv-AE from trained model
        elif flagProcess[kk] == 6:        

            fileAEModel = './ResSSTNATL60/model_patchDataset_NATL60withMETOP_SST_128_512_011WFilter005_NFilter100_RU003_Torch_Alpha101_AETRwoTrueData07D20N00W00_Nproj01_Grad_00_01_10_modelAEGradFP_iter015.mod'
            #'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj05_Encoder_iter006.mod'#'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj10_Encoder_iter011.mod'

            # PCA decomposition for comparison
            pca              = decomposition.PCA(DimCAE)
            pca.fit(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])))
                        
            
            rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
            rec_PCA_Tt[:,DimCAE:] = 0.
            rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
            mse_PCA_Tt       = np.mean( (rec_PCA_Tt - x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))**2 )
            var_Tt           = np.mean( (x_test-np.mean(x_train,axis=0))** 2 )
            exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt
            
            print(".......... PCA Dim = %d"%(DimCAE))
            print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
            print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))
        
            # Modify/Check data format
            x_train         = np.moveaxis(x_train, -1, 1)
            x_train_missing = np.moveaxis(x_train_missing, -1, 1)
            mask_train      = np.moveaxis(mask_train, -1, 1)

            x_test         = np.moveaxis(x_test, -1, 1)
            mask_test      = np.moveaxis(mask_test, -1, 1)
            x_test_missing = np.moveaxis(x_test_missing, -1, 1)
            print("... Training datashape: "+str(x_train.shape))
            print("... Test datashape    : "+str(x_test.shape))

            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            stdTr    = np.std( x_train )
            stdTt    = np.std( x_test )

            ## instantiate model for GPU implementation
            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cuda")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
    
    
            NBProjCurrent = 1
            NBGradCurrent = 10
            print('..... AE Model type : %d '%(flagAEType))
            print('..... Gradient type : %d '%(flagGradModel))
            print('..... Optim type    : %d '%(flagOptimMethod))
            print('..... DinAE configuration: NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
            
            #Model visualisation
            inputs = torch.randn(21,11,128,512)
            y = model_AE.encoder(torch.autograd.Variable(inputs))
            print(y.size())
            
            inputs = torch.randn(21,20,4,16)
            y = model_AE.decoder(torch.autograd.Variable(inputs))
            print(y.size())

            # NiterProjection,NiterGrad: global variables
            # bug for NiterProjection = 0
            shapeData       = x_train.shape[1:]
            #model_AE_GradFP = Model_AE_GradFP(model_AE2,shapeData,NiterProjection,NiterGrad,GradType,OptimType)
            model = dinAE.Model_AE_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,flagGradModel,flagOptimMethod)
        
            model = model.to(device)

            # load Model
            print('.......... Load trained model:'+fileAEModel)
            model.load_state_dict(torch.load(fileAEModel))
                                                    
            iterInit    = 0
            comptUpdate = 0

            ## Performance summary for best model
            # Daloader during training phase                
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_missing),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_missing),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }
            
            
            ## ouputs for training data
            x_train_pred = []
            for inputs_missing,masks,inputs_GT in dataloaders['train']:
              inputs_missing = inputs_missing.to(device)
              masks          = masks.to(device)
              inputs_GT      = inputs_GT.to(device)
              with torch.set_grad_enabled(True): 
              #with torch.set_grad_enabled(phase == 'train'):
                  outputs_ = model(inputs_missing,masks)
              if len(x_train_pred) == 0:
                  x_train_pred  = torch.mul(1.0,outputs_).cpu().detach()
              else:
                  x_train_pred  = np.concatenate((x_train_pred,torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

            ## ouputs for test data
            x_test_pred = []
            for inputs_missing,masks,inputs_GT in dataloaders['val']:
              inputs_missing = inputs_missing.to(device)
              masks          = masks.to(device)
              inputs_GT      = inputs_GT.to(device)
              with torch.set_grad_enabled(True): 
              #with torch.set_grad_enabled(phase == 'train'):
                  outputs_ = model(inputs_missing,masks)
              if len(x_test_pred) == 0:
                  x_test_pred  = torch.mul(1.0,outputs_).cpu().detach().numpy()
              else:
                  x_test_pred  = np.concatenate((x_test_pred,torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

            mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp = eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                                       mask_test,x_test,x_test_missing,x_test_pred)
            
            print(".......... Interpolation/reconstruction performance ")
            print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
            print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
            print('....')
            print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
            print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
            print('....')
            print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
            print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))
                    
            # AE performance of the trained AE applied to gap-free data
             ## ouputs for training data
            rec_AE_Tr = []
            for inputs_missing,masks,inputs_GT in dataloaders['train']:
              inputs_missing = inputs_missing.to(device)
              masks          = masks.to(device)
              inputs_GT      = inputs_GT.to(device)
              with torch.set_grad_enabled(True): 
              #with torch.set_grad_enabled(phase == 'train'):
                  outputs_ = model.model_AE(inputs_GT)
              if len(rec_AE_Tr) == 0:
                  rec_AE_Tr  = torch.mul(1.0,outputs_).cpu().detach()
              else:
                  rec_AE_Tr  = np.concatenate((rec_AE_Tr,torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

            rec_AE_Tt = []
            for inputs_missing,masks,inputs_GT in dataloaders['val']:
              inputs_missing = inputs_missing.to(device)
              masks          = masks.to(device)
              inputs_GT      = inputs_GT.to(device)
              with torch.set_grad_enabled(True): 
              #with torch.set_grad_enabled(phase == 'train'):
                  outputs_ = model.model_AE(inputs_GT)
              if len(rec_AE_Tt) == 0:
                  rec_AE_Tt  = torch.mul(1.0,outputs_).cpu().detach().numpy()
              else:
                  rec_AE_Tt  = np.concatenate((rec_AE_Tt,torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

            exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
            
            print(".......... Auto-encoder performance when applied to gap-free data")
            print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
            print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))
                    
