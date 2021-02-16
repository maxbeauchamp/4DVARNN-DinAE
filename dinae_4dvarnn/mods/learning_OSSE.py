from dinae_4dvarnn import *
from tools import *
from graphics import *
from eval_Performance      import eval_AEPerformance
from eval_Performance      import eval_InterpPerformance
from plot_Figs             import plot_Figs
from save_Pickle           import save_Pickle
from save_NetCDF           import save_NetCDF
from save_Models           import save_Models
from Model_4DVarNN_FP      import Model_4DVarNN_FP
from Model_4DVarNN_Grad    import Model_4DVarNN_Grad
from Model_4DVarNN_GradFP  import Model_4DVarNN_GradFP

def add_covariates_to_tensor(tensor1,tensor2,N_cov):
    new_tensor = np.zeros(tensor2.shape)
    index      = np.arange(0,tensor2.shape[1],N_cov+1)
    new_tensor[:,index,:,:] = tensor1.cpu().detach().numpy()
    index2      = np.delete(range(tensor2.shape[1]),index)
    new_tensor[:,index2,:,:] = (tensor2.cpu().detach().numpy())[:,index2,:,:]
    return torch.Tensor(new_tensor)

class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def model_to_MultiGPU(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('... Number of GPUs: %d'%torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model      = MyDataParallel(model)
            model.model_AE   = MyDataParallel(model.model_AE)
            model.model_Grad = MyDataParallel(model.model_Grad)
    model.to(device)
    model.model_AE.to(device)
    model.model_Grad.to(device)
    return model

def learning_OSSE(dict_global_Params,genFilename,\
                  input_train,mask_train,target_train,input_train_OI,lday_pred,meanTr,stdTr,\
                  input_test,mask_test,target_test,input_test_OI,lday_test,model_AE,DIMCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    # ***************** #
    # model compilation #
    # ***************** #

    ## model parameters
    if solver_type=='FP':
        #NbProjection   = [2,4,5,6,7,9,10]
        #NbProjection   = [5,5,5,5,5,5,5]
        NbProjection   = [1,1,2,3,4,5,5]
        NbGradIter     = [0,0,0,0,0,0,0]
    else:
        NbProjection   = [0,0,0,0,0,0,0]
        #NbProjection   = [1,2,3,4,4,5,5]
        #NbProjection   = [2,4,5,6,7,9,10]
        #NbGradIter     = [1,2,3,4,4,5,5]
        #NbProjection   = [2,2,2,2,2,2,2]
        NbGradIter     = [1,2,3,4,4,5,5]
        #NbGradIter     = [2,2,2,5,5,5,5]
        #NbGradIter    = [0,0,0,0,0,0,0]
    if flagTrWMissingData==2:
        #lrUpdate       = [1e-4,1e-4,1e-5,1e-5,1e-6,1e-6,1e-7]
        lrUpdate       = [1e-2,1e-3,1e-4,1e-5,1e-5,1e-6,1e-6]
    else:
        lrUpdate       = [1e-3,1e-4,1e-4,1e-5,1e-5,1e-6,1e-6]
    IterUpdate     = [0,1,2,3,4,5,6]
    val_split      = 0.1
    comptUpdate    = 0  

    ## modify/check data format
    input_train     = np.moveaxis(input_train, -1, 1)
    mask_train      = np.moveaxis(mask_train, -1, 1)
    target_train    = np.moveaxis(target_train, -1, 1)
    input_test      = np.moveaxis(input_test, -1, 1)
    mask_test       = np.moveaxis(mask_test, -1, 1)
    target_test     = np.moveaxis(target_test, -1, 1)

    # Replace NaN value with zeros
    input_train  = np.nan_to_num(input_train)
    target_train = np.nan_to_num(target_train)
    input_test   = np.nan_to_num(input_test)
    target_test  = np.nan_to_num(target_test)

    # first initialize the solution
    input_train_init    = input_train
    input_test_init     = input_test
 
    print("... Training datashape: "+str(input_train.shape))
    print("... Test datashape    : "+str(input_test.shape))
    ## define dataloaders with randomised batches (no random shuffling for validation/test data)
    training_dataset = torch.utils.data.TensorDataset(\
                            torch.Tensor(input_train_init),\
                            torch.Tensor(mask_train),\
                            torch.Tensor(target_train)) 
    test_dataset     = torch.utils.data.TensorDataset(\
                            torch.Tensor(input_test_init),\
                            torch.Tensor(mask_test),\
                            torch.Tensor(target_test))       
    dataset_sizes = {'train': len(training_dataset), 'val': len(test_dataset)} 
    ## instantiate model for GPU implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(".... Device GPU: "+str(torch.cuda.is_available()))
    ## initialize or load the model (bug for number of FP iterations = 0) 
    shapeData       = input_train.shape[1:]  
    NBProjCurrent = NbProjection[0]
    NBGradCurrent = NbGradIter[0]
    print('..... DinAE learning (initialisation): NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
    if flagLoadModel == 1:
        model.load_state_dict(torch.load(fileAEModelInit))
    else:
        model = Model_4DVarNN_GradFP(\
              model_AE,shapeData,NBProjCurrent,NBGradCurrent,\
              flagGradModel,flagOptimMethod,N_cov=N_cov)    
    model =  model_to_MultiGPU(model)
    ## create an optimizer object (Adam with lr 1e-3)
    lrCurrent        = lrUpdate[0]
    lambda_LRAE = 0.5
    optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},\
                              {'params': model.model_AE.encoder.parameters(),\
                               'lr': lambda_LRAE*lrCurrent}
                              ], lr=lrCurrent)
    '''optimizer   = optim.RMSprop([{'params': model.model_Grad.parameters()},\
                              {'params': model.model_AE.encoder.parameters(),\
                               'lr': lambda_LRAE*lrCurrent}
                              ], lr=lrCurrent)
    '''

    ## adapt loss function parameters if learning only with observations
    if flagTrWMissingData == 2:
        model.model_Grad.compute_Grad.alphaObs = torch.nn.Parameter(torch.Tensor([np.sqrt(alpha4DVar[0])]).to(device))
        model.model_Grad.compute_Grad.alphaAE  = torch.nn.Parameter(torch.Tensor([np.sqrt(alpha4DVar[1])]).to(device))
        model.model_Grad.compute_Grad.alphaObs.requires_grad = False
        model.model_Grad.compute_Grad.alphaAE.requires_grad  = False
        alpha_Grad = alpha4DVar[0]
        #alpha_FP   = 1. - alpha[0]
        alpha_AE   = alpha4DVar[1]
    else:
        alpha_Grad = alpha[0]
        #alpha_FP   = 1. - alpha[0]
        alpha_AE   = alpha[1]
    
    ## model compilation 
    since = time.time()
    best_model_wts   = copy.deepcopy(model.state_dict())

    # ******************** #
    # Start Learning model #
    # ******************** #
        
    print("..... Start learning AE model %d FP/Grad %d"%(flagAEType,flagOptimMethod))
    best_loss      = 1e10
    iter = 0
    while iter<Niter:
        print('######################')
        print('ITERATION %d, LR %e'%(iter,lrCurrent))
        print('######################')
        ## update number of FP projections, number of GB iterations, learning rate and the corresponding model
        if iter == IterUpdate[comptUpdate]:
            NBProjCurrent = NbProjection[comptUpdate]
            NBGradCurrent = NbGradIter[comptUpdate]
            lrCurrent     = lrUpdate[comptUpdate]
            print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))
            # update GradFP architectures
            print('..... Update model architecture')
            model = Model_4DVarNN_GradFP(\
                      model_AE,shapeData,NBProjCurrent,NBGradCurrent,\
                      flagGradModel,flagOptimMethod,N_cov=N_cov)
            model =  model_to_MultiGPU(model)
            # update optimizer
            optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},
                                    {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                    ], lr=lrCurrent)
            '''optimizer   = optim.RMSprop([{'params': model.model_Grad.parameters()},
                                    {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                    ], lr=lrCurrent)
            '''
            # copy model parameters from current model
            model.load_state_dict(best_model_wts)                            
            # update comptUpdate
            if comptUpdate < len(NbProjection)-1:
                comptUpdate += 1
        ## daloader for the training phase                
        dataloaders = { 'train': torch.utils.data.DataLoader(training_dataset,\
                                 batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),\
                        'val':   torch.utils.data.DataLoader(test_dataset,\
                                 batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True), }

        ## run NbEpoc training epochs
        best_loss_epoch      = 1e10
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
                for inputs_init, masks, targets in dataloaders[phase]:
                    compt = compt+1
                    inputs_init    = inputs_init.to(device)
                    masks          = masks.to(device)
                    index          = np.arange(0,masks.shape[1],N_cov+1)
                    masks_inputs   = masks[:,index,:,:]
                    targets        = targets.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # need to evaluate grad/backward during the evaluation and training phase for model_AE
                    with torch.set_grad_enabled(True): 
                        inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
                        if model.OptimType == 1:
                            outputs,grad_new,normgrad = model(inputs_init,masks,None)
                        elif model.OptimType == 2:
                            outputs,hidden_new,cell_new,normgrad = model(inputs_init,masks,None,None)
                        else:
                            outputs,normgrad = model(inputs_init,masks)

                        ## Losses
                        # full DAW
                        if flagTrWMissingData==2:
                            idT1 = np.arange(0,inputs_init.shape[1],N_cov+1)
                            idT2 = np.arange(0,targets.shape[1],1)
                        # on center image of the DAW
                        else:
                            idT1 = int(np.floor(inputs_init.shape[1]/2))
                            idT2 = int(np.floor(targets.shape[1]/2))

                        # compute losses
                        loss_R      = torch.sum((outputs[:,idT2,:,:] - targets[:,idT2,:,:])**2 *masks_inputs [:,idT2,:,:])
                        loss_R      = torch.mul(1.0 / torch.sum(masks_inputs[:,idT2,:,:]),loss_R)
                        loss_I      = torch.sum((outputs[:,idT2,:,:] - targets[:,idT2,:,:])**2 * (1. - masks_inputs[:,idT2,:,:]) )
                        loss_I      = torch.mul(1.0 / torch.sum(1.-masks_inputs[:,idT2,:,:]),loss_I)
                        loss_All    = torch.mean((outputs[:,idT2,:,:] - targets[:,idT2,:,:])**2 )
                        if N_cov>0:
                            outputs_wcov = add_covariates_to_tensor(outputs,inputs_init,N_cov).to(device) 
                            targets_wcov = add_covariates_to_tensor(targets,inputs_init,N_cov).to(device)
                        else:
                            outputs_wcov = outputs
                            targets_wcov = targets
                        loss_AE     = torch.mean((model.model_AE(outputs_wcov)[:,idT2,:,:] - outputs[:,idT2,:,:])**2 )
                        loss_AE_GT  = torch.mean((model.model_AE(targets_wcov)[:,idT2,:,:] - targets[:,idT2,:,:])**2 )
                        index      = np.arange(0,inputs_init.shape[1],N_cov+1)
                        loss_Obs    = torch.sum( (outputs[:,idT2,:,:] - inputs_init[:,idT1,:,:])**2 * masks_inputs[:,idT2,:,:] )
                        loss_Obs    = loss_Obs / torch.sum( masks_inputs[:,idT2,:,:] )
                        if flagTrWMissingData == 2:
                            spatial_gradients_avg = einops.reduce(sobel(outputs), 'b t lat lon -> 1', 'mean')
                            loss        = alpha4DVar[0] * loss_Obs + alpha4DVar[1] * loss_AE + spatial_gradients_avg
                        else:
                            #spatial_gradients_avg = einops.reduce(sobel(torch.unsqueeze(outputs[:,idT2,:,:],1)), 'b t lat lon -> 1', 'mean')
                            loss_Grad   = torch.sqrt(einops.reduce((sobel(torch.unsqueeze(outputs[:,idT2,:,:],1))-sobel(torch.unsqueeze(targets[:,idT2,:,:],1)))**2, 'b t lat lon -> 1', 'mean'))
                            loss        = loss_All + 10*loss_Grad + .1*loss_AE + .1*loss_AE_GT
                            #loss        = loss_All + loss_Grad
                         
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                        # statistics
                        running_loss             += loss.item() * inputs_init.size(0)
                        running_loss_I           += loss_I.item() * inputs_init.size(0)
                        running_loss_R           += loss_R.item() * inputs_init.size(0)
                        running_loss_All         += loss_All.item() * inputs_init.size(0)
                        running_loss_AE          += loss_AE_GT.item() * inputs_init.size(0)
                        num_loss                 += inputs_init.size(0)
    
                    epoch_loss       = running_loss / num_loss
                    epoch_loss_All   = running_loss_All / num_loss
                    epoch_loss_AE    = running_loss_AE / num_loss
                    epoch_loss_I     = running_loss_I / num_loss
                    epoch_loss_R     = running_loss_R / num_loss
                        
                    if not isinstance(stdTr, list) :
                        meanTr=[meanTr]
                        stdTr=[stdTr]

                    epoch_nloss_All = epoch_loss_All / stdTr[0]**2
                    epoch_nloss_I   = epoch_loss_I / stdTr[0]**2
                    epoch_nloss_R   = epoch_loss_R / stdTr[0]**2
                    epoch_nloss_AE  = loss_AE / stdTr[0]**2
            
                    # deep copy the model
                    if phase == 'val' and epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                    # deep copy the model
                    if phase == 'val' and epoch_loss < best_loss_epoch:
                        best_loss_epoch = epoch_loss
                
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                  time_elapsed // 60, time_elapsed % 60))
                print('Best val loss: {:4f}'.format(best_loss))
                print('Best val loss epoch: {:4f}'.format(best_loss_epoch))

        # ********************************** #
        # Prediction on training & test data #
        # ********************************** #

        ## load best model weights
        model.load_state_dict(best_model_wts)

        ## AE performance on training and validation datasets
        # outputs for training data
        x_train_pred = []
        for inputs_init,masks,targets in dataloaders['train']:
            inputs_init    = inputs_init.to(device)
            masks          = masks.to(device)
            targets     = targets.to(device)
            with torch.set_grad_enabled(True): 
                outputs_ = model(inputs_init,masks)[0]
            if len(x_train_pred) == 0:
                x_train_pred  = torch.mul(1.0,outputs_).cpu().detach()
            else:
                x_train_pred  = np.concatenate((x_train_pred,\
                                   torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)
        # outputs for test data
        x_test_pred = []
        for inputs_init, masks,targets in dataloaders['val']:
            inputs_init    = inputs_init.to(device)
            masks          = masks.to(device)
            targets     = targets.to(device)
            with torch.set_grad_enabled(True): 
                outputs_ = model(inputs_init,masks)[0]
            if len(x_test_pred) == 0:
                x_test_pred  = torch.mul(1.0,outputs_).cpu().detach().numpy()
            else:
                x_test_pred  = np.concatenate((x_test_pred,\
                                 torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

        ## AE performance of the trained AE applied to gap-free data
        # ouputs for training data
        rec_AE_Tr = []
        for inputs_init,masks,targets in dataloaders['train']:
            inputs_init    = inputs_init.to(device)
            masks          = masks.to(device)
            targets        = targets.to(device)
            if N_cov>0:
                targets_wcov = add_covariates_to_tensor(targets,\
                                inputs_init,N_cov).to(device)
            else:
                targets_wcov = targets
            with torch.set_grad_enabled(True): 
                outputs_ = model.model_AE(targets_wcov)
            if len(rec_AE_Tr) == 0:
                rec_AE_Tr  = torch.mul(1.0,outputs_).cpu().detach()
            else:
                rec_AE_Tr  = np.concatenate((rec_AE_Tr,\
                               torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

        # ouputs for test data
        rec_AE_Tt = []
        for inputs_init,masks,targets in dataloaders['val']:
            inputs_init    = inputs_init.to(device)
            masks          = masks.to(device)
            targets      = targets.to(device)
            if N_cov>0:
                targets_wcov = add_covariates_to_tensor(targets,\
                                inputs_init,N_cov).to(device)
            else:
                targets_wcov = targets
            with torch.set_grad_enabled(True): 
                outputs_ = model.model_AE(targets_wcov)
            if len(rec_AE_Tt) == 0:
                rec_AE_Tt  = torch.mul(1.0,outputs_).cpu().detach().numpy()
            else:
                rec_AE_Tt  = np.concatenate((rec_AE_Tt,torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

        index = np.arange(0,input_train.shape[1],N_cov+1)
        mse_train,exp_var_train,\
        mse_test,exp_var_test,\
        mse_train_interp,exp_var_train_interp,\
        mse_test_interp,exp_var_test_interp =\
        eval_InterpPerformance(mask_train[:,index,:,:],target_train,x_train_pred,\
                               mask_test[:,index,:,:],target_test,x_test_pred)
        exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(input_train[:,index,:,:],rec_AE_Tr,input_test[:,index,:,:],rec_AE_Tt)
        
        if not isinstance(stdTr, list) :
            meanTr=[meanTr]
            stdTr=[stdTr]
        print(".......... iter %d"%(iter))
        print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr[0]**2,100.*exp_var_train[1]))
        print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr[0]**2,100.*exp_var_test[1]))
        print('....')
        print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr[0]**2,100.*exp_var_train[0]))
        print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr[0]**2,100.*exp_var_test[0]))
        print('....')
        print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr[0]**2,100.*exp_var_train_interp))
        print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr[0]**2,100.*exp_var_test_interp))

        # **************************** #
        # Plot figures and Save models #
        # **************************** #

        ## save models
        genSuffixModel=save_Models(dict_global_Params,genFilename,alpha4DVar,
                                   NBProjCurrent,NBGradCurrent,model_AE,model,iter)

        ## generate some plots
        plot_Figs(dict_global_Params,genFilename,genSuffixModel,
                  target_train,input_train,x_train_pred,rec_AE_Tr,input_train_OI,meanTr[0],stdTr[0],
                  target_test,input_test,x_test_pred,rec_AE_Tt,input_test_OI,
                  lday_pred,lday_test,iter)

        ## save results in a pickle file
        save_Pickle(dict_global_Params,
                    target_train,input_train,x_train_pred,rec_AE_Tr,input_train_OI,meanTr[0],stdTr[0],
                    target_test,input_test,x_test_pred,rec_AE_Tt,input_test_OI,iter)

        ## save results in NetCDF file
        save_NetCDF(dict_global_Params,meanTr[0],stdTr[0],
                    target_test,input_test,x_test_pred,rec_AE_Tt,input_test_OI,
                    iter)

        '''# if nothing happened during this iteration
        if best_loss_epoch > best_loss:
            # copy model parameters from best model
            model.load_state_dict(best_model_wts)
            # update comptUpdate
            comptUpdate -= 1
            lrCurrent   /=10
            iter        -=1
            lrUpdate    = [lr/10. for lr in lrUpdate]
        else:
            iter        +=1'''
        iter+=1

