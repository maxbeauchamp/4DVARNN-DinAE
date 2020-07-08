from 4dvarnn-dinae import *
from .utils.tools import *
from .utils.graphics import *
from .utils.eval_Performance      import eval_AEPerformance
from .utils.eval_Performance      import eval_InterpPerformance
from .utils.plot_Figs             import plot_Figs
from .utils.save_Models           import save_Models
sys.path.insert(0,'/linkhome/rech/genimt01/uba22to/4DVARNN-DinAE/4dvarnn-dinae/mods')

from .utils.utils_solver.Model_4DVarNN_FP      import Model_4DVarNN_FP
from .utils.utils_solver.Model_4DVarNN_Grad    import Model_4DVarNN_GradFP
from .utils.utils_solver.Model_4DVarNN_GradFP  import Model_4DVarNN_GradFP

def learning_OSSE(dict_global_Params,genFilename,x_train,x_train_missing,mask_train,gt_train,\
                        meanTr,stdTr,x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,model_AE,DimCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    # ******************************** #
    # PCA decomposition for comparison #
    # *********************************#

    # train PCA
    pca      = decomposition.PCA(DimCAE)
    pca.fit(np.reshape(gt_train,(gt_train.shape[0],gt_train.shape[1]*gt_train.shape[2]*gt_train.shape[3])))
    
    # apply PCA to test data
    rec_PCA_Tt       = pca.transform(np.reshape(gt_test,(gt_test.shape[0],gt_test.shape[1]*gt_test.shape[2]*gt_test.shape[3])))
    rec_PCA_Tt[:,DimCAE:] = 0.
    rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
    mse_PCA_Tt       = np.mean( (rec_PCA_Tt - gt_test.reshape((gt_test.shape[0],gt_test.shape[1]*gt_test.shape[2]*gt_test.shape[3])))**2 )
    var_Tt           = np.mean( (gt_test-np.mean(gt_train,axis=0))** 2 )
    exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt
    
    print(".......... PCA Dim = %d"%(DimCAE))
    print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
    print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))

    print("..... Regularization parameters: dropout = %.3f, wl2 = %.2E"%(dropout,wl2))
    
    # ***************** #
    # model compilation #
    # ***************** #

    # model fit
    NbProjection   = [0,0,2,2,5,5,10,15,14]
    NbProjection   = [2,5,5,5]
    NbGradIter     = [0,2,5,5,10,10,15]
    lrUpdate       = [1e-3,1e-4,1e-5,1e-5,1e-5,1e-6,1e-6,1e-5,1e-6]
    lrUpdate       = [1e-4,1e-5,1e-6,1e-7]
    #lrUpdate       = [1e-3,1e-4,1e-5,1e-6]
    IterUpdate     = [0,3,10,15,20,25,30,35,40]
    #IterUpdate     = [0,6,15,20]
    val_split      = 0.1
    
    iterInit = 0
    comptUpdate   = 0  

    # Modify/Check data format
    x_train         = np.moveaxis(x_train, -1, 1)
    x_train_missing = np.moveaxis(x_train_missing, -1, 1)
    mask_train      = np.moveaxis(mask_train, -1, 1)
    gt_train        = np.moveaxis(gt_train, -1, 1)
    x_test          = np.moveaxis(x_test, -1, 1)
    mask_test       = np.moveaxis(mask_test, -1, 1)
    x_test_missing  = np.moveaxis(x_test_missing, -1, 1)
    gt_test         = np.moveaxis(gt_test, -1, 1)
    print("... Training datashape: "+str(x_train.shape))
    print("... Test datashape    : "+str(x_test.shape))

    # mean-squared error loss
    stdTr    = np.std( x_train )
    stdTt    = np.std( x_test )
    print()
    print('....   stdTr = %.3f'%stdTr)
    print('....   stdTt = %.3f'%stdTt)

    ## Define dataloaders with randomised batches     
    ## no random shuffling for validation/test data
    training_dataset     = torch.utils.data.TensorDataset(\
                            torch.Tensor(x_train_missing),\
                            torch.Tensor(mask_train),\
                            torch.Tensor(gt_train)) # create your dataset
    test_dataset         = torch.utils.data.TensorDataset(\
                            torch.Tensor(x_test_missing),\
                            torch.Tensor(mask_test),\
                            torch.Tensor(gt_test)) # create your datset                    
    dataset_sizes = {'train': len(training_dataset), 'val': len(test_dataset)} 

    ## instantiate model for GPU implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(".... Device GPU: "+str(torch.cuda.is_available()))


    NBProjCurrent = NbProjection[0]
    NBGradCurrent = NbGradIter[0]
    print('..... DinAE learning (initialisation): NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
    #Model visualisation
    inputs = torch.randn(21,11,200,200)
    y = model_AE.encoder(torch.autograd.Variable(inputs))
    print(y.size())            
    inputs = torch.randn(21,200,4,4)
    y = model_AE.decoder(torch.autograd.Variable(inputs))
    print(y.size())
    # NiterProjection, NiterGrad: global variables
    # bug for NiterProjection = 0
    shapeData       = x_train.shape[1:]  
    flagLoadModelAE = 0
    if flagLoadModelAE == 1:
        model.load_state_dict(torch.load(fileAEModelInit))
    else:
        model = Model_4DVarNN_GradFP(\
              model_AE,shapeData,NBProjCurrent,NBGradCurrent,\
              flagGradModel,flagOptimMethod)    
    # create an optimizer object (Adam with lr 1e-3)
    lrCurrent        = lrUpdate[0]
    optimizer        = optim.Adam(model.parameters(), lr=lrCurrent)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.)
                        
    # model compilation 
    since = time.time()
    alpha_MaskedLoss = alpha_Losss[0]
    alpha_GTLoss     = 1. - alpha_Losss[0]
    alpha_AE         = alpha_Losss[1]
    best_model_wts   = copy.deepcopy(model.state_dict())

    # ******************** #
    # Start Learning model #
    # ******************** #
        
    print("..... Start learning AE model %d FP/Grad %d"%(flagAEType,flagOptimMethod))
    best_loss      = 1e10
    for iter in range(iterInit,Niter):
        if iter == IterUpdate[comptUpdate]:
            # update GradFP parameters
            NBProjCurrent = NbProjection[comptUpdate]
            NBGradCurrent = NbGradIter[comptUpdate]
            lrCurrent     = lrUpdate[comptUpdate]
            print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))
            # update GradFP architectures
            print('..... Update model architecture')
            model = Model_4DVarNN_GradFP(\
                      model_AE,shapeData,NBProjCurrent,NBGradCurrent,\
                      flagGradModel,flagOptimMethod)
            model = model.to(device)        
            # update optimizer
            optimizer        = optim.Adam(model.parameters(), lr= lrCurrent)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
            # copy model parameters from current model
            model.load_state_dict(best_model_wts)                            
            # update comptUpdate
            if comptUpdate < len(NbProjection)-1:
                comptUpdate += 1

        # Daloader during training phase                
        dataloaders = { 'train': torch.utils.data.DataLoader(training_dataset,\
                                   batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),\
                        'val': torch.utils.data.DataLoader(test_dataset,\
                                   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                      }
                
        # Run NbEpoc training epochs
        for epoch in range(NbEpoc):
            print('Epoc %d/%d'%(epoch,NbEpoc))
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train()  # Set model to training mode
                    print('..... Training step')
                else:
                    model.eval()   # Set model to evaluate mode
                    print('..... Test step')
            
                running_loss         = 0.0
                running_loss_All     = 0.
                running_loss_R       = 0.
                running_loss_I       = 0.
                running_loss_AE      = 0.
                num_loss     = 0
    
                # Iterate over data.
                compt = 0
                for inputs_missing,masks,inputs_GT in dataloaders[phase]:
                    compt = compt+1
                    inputs_missing = inputs_missing.to(device)
                    masks          = masks.to(device)
                    inputs_GT      = inputs_GT.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # need to evaluate grad/backward during the evaluation and training phase for model_AE
                    with torch.set_grad_enabled(True): 
                        outputs = model(inputs_missing,masks)
                        loss_R      = torch.sum((outputs - inputs_GT)**2 * masks )
                        loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
                        loss_I      = torch.sum((outputs - inputs_GT)**2 * (1. - masks) )
                        loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
                        loss_All    = torch.mean((outputs - inputs_GT)**2 )
                        loss_AE     = torch.mean((model.model_AE(outputs) - outputs)**2 )
                        loss_AE_GT  = torch.mean((model.model_AE(inputs_GT) - inputs_GT)**2 )

                        if alpha_MaskedLoss > 0.:
                            loss = torch.mul(alpha_MaskedLoss,loss_R)
                        else: 
                            loss = torch.mul(alpha_GTLoss,loss_All)
                            loss = torch.add(loss,torch.mul(alpha_AE,loss_AE))
                                
                        if phase == 'train':
                            loss = torch.mul(1.,loss_All)
                        else :
                            loss = torch.mul(1.,loss_R)
                        loss = torch.add(loss,torch.mul(alpha_AE,loss_AE))

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss             += loss.item() * inputs_missing.size(0)
                    running_loss_I           += loss_I.item() * inputs_missing.size(0)
                    running_loss_R           += loss_R.item() * inputs_missing.size(0)
                    running_loss_All         += loss_All.item() * inputs_missing.size(0)
                    running_loss_AE          += loss_AE_GT.item() * inputs_missing.size(0)
                    num_loss                 += inputs_missing.size(0)
    
                if phase == 'train':
                    exp_lr_scheduler.step()
    
                epoch_loss       = running_loss / num_loss
                epoch_loss_All   = running_loss_All / num_loss
                epoch_loss_AE    = running_loss_AE / num_loss
                epoch_loss_I     = running_loss_I / num_loss
                epoch_loss_R     = running_loss_R / num_loss
                        
                if phase == 'train':
                    epoch_nloss_All = epoch_loss_All / stdTr**2
                    epoch_nloss_I   = epoch_loss_I / stdTr**2
                    epoch_nloss_R   = epoch_loss_R / stdTr**2
                    epoch_nloss_AE  = loss_AE / stdTr**2
                else:
                    epoch_nloss_All = epoch_loss_All / stdTt**2
                    epoch_nloss_I   = epoch_loss_I / stdTt**2
                    epoch_nloss_R   = epoch_loss_R / stdTt**2
                    epoch_nloss_AE   = loss_AE / stdTt**2
            
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                  time_elapsed // 60, time_elapsed % 60))
            print('Best val loss: {:4f}'.format(best_loss))


        # ********************************** #
        # Prediction on training & test data #
        # ********************************** #

        ## load best model weights
        model.load_state_dict(best_model_wts)

        ## Performance summary for best model
        # Daloader during training phase                
        dataloaders = { 'train': torch.utils.data.DataLoader(training_dataset,\
                                   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),\
                        'val': torch.utils.data.DataLoader(test_dataset,\
                                   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                      }

        ## outputs for training data
        x_train_pred = []
        for inputs_missing,masks,inputs_GT in dataloaders['train']:
            inputs_missing = inputs_missing.to(device)
            masks          = masks.to(device)
            inputs_GT      = inputs_GT.to(device)
            with torch.set_grad_enabled(True): 
                outputs_ = model(inputs_missing,masks)
            if len(x_train_pred) == 0:
                x_train_pred  = torch.mul(1.0,outputs_).cpu().detach()
            else:
                x_train_pred  = np.concatenate((x_train_pred,\
                                   torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)
        ## outputs for test data
        x_test_pred = []
        for inputs_missing,masks,inputs_GT in dataloaders['val']:
            inputs_missing = inputs_missing.to(device)
            masks          = masks.to(device)
            inputs_GT      = inputs_GT.to(device)
            with torch.set_grad_enabled(True): 
                outputs_ = model(inputs_missing,masks)
            if len(x_test_pred) == 0:
                x_test_pred  = torch.mul(1.0,outputs_).cpu().detach().numpy()
            else:
                x_test_pred  = np.concatenate((x_test_pred,\
                                 torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

        ## AE performance of the trained AE applied to gap-free data
        # ouputs for training data
        rec_AE_Tr = []
        for inputs_missing,masks,inputs_GT in dataloaders['train']:
            inputs_missing = inputs_missing.to(device)
            masks          = masks.to(device)
            inputs_GT      = inputs_GT.to(device)
            with torch.set_grad_enabled(True): 
                outputs_ = model.model_AE(inputs_GT)
            if len(rec_AE_Tr) == 0:
                rec_AE_Tr  = torch.mul(1.0,outputs_).cpu().detach()
            else:
                rec_AE_Tr  = np.concatenate((rec_AE_Tr,\
                               torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)
        # ouputs for test data
        rec_AE_Tt = []
        for inputs_missing,masks,inputs_GT in dataloaders['val']:
            inputs_missing = inputs_missing.to(device)
            masks          = masks.to(device)
            inputs_GT      = inputs_GT.to(device)
            with torch.set_grad_enabled(True): 
                outputs_ = model.model_AE(inputs_GT)
            if len(rec_AE_Tt) == 0:
                rec_AE_Tt  = torch.mul(1.0,outputs_).cpu().detach().numpy()
            else:
                rec_AE_Tt  = np.concatenate((rec_AE_Tt,torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

        exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
        print(".......... Auto-encoder performance when applied to gap-free data")
        print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
        print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))

        # remove additional covariates from variables
        if include_covariates == True:
            mask_train_wc, x_train_wc, x_train_init_wc, x_train_missing_wc,\
            mask_test_wc, x_test_wc, x_test_init_wc, x_test_missing_wc,\
            meanTr_wc, stdTr_wc=\
            mask_train, x_train, x_train_init, x_train_missing,\
            mask_test, x_test, x_test_init, x_test_missing,\
            meanTr, stdTr
            index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
            mask_train      = mask_train[:,index,:,:]
            x_train         = x_train[:,index,:,:]
            x_train_init    = x_train_init[:,index,:,:]
            x_train_missing = x_train_missing[:,index,:,:]
            mask_test      = mask_test[:,index,:,:]
            x_test         = x_test[:,index,:,:]
            x_test_init    = x_test_init[:,index,:,:]
            x_test_missing = x_test_missing[:,index,:,:]
            meanTr = meanTr[0]
            stdTr  = stdTr[0]

        mse_train,exp_var_train,\
        mse_test,exp_var_test,\
        mse_train_interp,exp_var_train_interp,\
        mse_test_interp,exp_var_test_interp =\
        eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,\
                               mask_test,x_test,x_test_missing,x_test_pred)
        
        print(".......... iter %d"%(iter))
        print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
        print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
        print('....')
        print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
        print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
        print('....')
        print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
        print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))


        # **************************** #
        # Plot figures and Save models #
        # **************************** #

        # save models
        genSuffixModel=save_Models(dict_global_Params,genFilename,alpha_Losss,\
                                   NBProjCurrent,NBGradCurrent,model_AE,model,iter)
 
        idT = int(np.floor(x_test.shape[1]/2))
        saved_path = dirSAVE+'/saved_path_%03d'%(iter)+'_FP_'+suf1+'_'+suf2+'.pickle'
        if flagloadOIData == 1:
            # generate some plots
            plot_Figs(dirSAVE,genFilename,genSuffixModel,\
                  (np.moveaxis(gt_train,1,3)*stdTr)+meanTr+x_train_OI,(np.moveaxis(x_train_missing,1,3)*stdTr)+meanTr+x_train_OI,np.moveaxis(mask_train,1,3),\
                  (np.moveaxis(x_train_pred,1,3)*stdTr)+meanTr+x_train_OI,(np.moveaxis(rec_AE_Tr,1,3)*stdTr)+meanTr+x_train_OI,\
                  (np.moveaxis(gt_test,1,3)*stdTr)+meanTr+x_test_OI,(np.moveaxis(x_test_missing,1,3)*stdTr)+meanTr+x_test_OI,np.moveaxis(mask_test,1,3),lday_test,\
                  (np.moveaxis(x_test_pred,1,3)*stdTr)+meanTr+x_test_OI,(np.moveaxis(rec_AE_Tt,1,3)*stdTr)+meanTr+x_test_OI,iter)
            # Save DINAE result         
            with open(saved_path, 'wb') as handle:
                pickle.dump([((np.moveaxis(gt_test,1,3)*stdTr)+meanTr+x_test_OI)[:,:,:,idT],((np.moveaxis(x_test_missing,1,3)*stdTr)+meanTr+x_test_OI)[:,:,:,idT],\
                         ((np.moveaxis(x_test_pred,1,3)*stdTr)+meanTr+x_test_OI)[:,:,:,idT],((np.moveaxis(rec_AE_Tt,1,3)*stdTr)+meanTr+x_test_OI)[:,:,:,idT]], handle)

        else:
            # generate some plots
            plot_Figs(dirSAVE,genFilename,genSuffixModel,\
                  (np.moveaxis(gt_train,1,3)*stdTr)+meanTr,(np.moveaxis(x_train_missing,1,3)*stdTr)+meanTr,np.moveaxis(mask_train,1,3),\
                  (np.moveaxis(x_train_pred,1,3)*stdTr)+meanTr,(np.moveaxis(rec_AE_Tr,1,3)*stdTr)+meanTr,\
                  (np.moveaxis(gt_test,1,3)*stdTr)+meanTr,(np.moveaxis(x_test_missing,1,3)*stdTr)+meanTr,np.moveaxis(mask_test,1,3),lday_test,\
                  (np.moveaxis(x_test_pred,1,3)*stdTr)+meanTr,(np.moveaxis(rec_AE_Tt,1,3)*stdTr)+meanTr,iter)
            # Save DINAE result         
            with open(saved_path, 'wb') as handle:
                pickle.dump([((np.moveaxis(gt_test,1,3)*stdTr)+meanTr)[:,:,:,idT],((np.moveaxis(x_test_missing,1,3)*stdTr)+meanTr)[:,:,:,idT],\
                         ((np.moveaxis(x_test_pred,1,3)*stdTr)+meanTr)[:,:,:,idT],((np.moveaxis(rec_AE_Tt,1,3)*stdTr)+meanTr)[:,:,:,idT]], handle)

        # reset variables with additional covariates
        if include_covariates == True:
            mask_train, x_train, x_train_init, x_train_missing,\
            mask_test, x_test, x_test_init, x_test_missing,\
            meanTr, stdTr=\
            mask_train_wc, x_train_wc, x_train_init_wc, x_train_missing_wc,\
            mask_test_wc, x_test_wc, x_test_init_wc, x_test_missing_wc,\
            meanTr_wc, stdTr_wc
