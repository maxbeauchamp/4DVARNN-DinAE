from dinae_4dvarnn import *
from sklearn import preprocessing
from tools import *
from graphics import *
from eval_Performance      import eval_AEPerformance2 as eval_AEPerformance
from eval_Performance      import eval_InterpPerformance2 as eval_InterpPerformance
from plot_Figs             import plot_Figs2 as plot_Figs
from save_Pickle           import save_Pickle2 as save_Pickle
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

def learning_OSE(dict_global_Params,genFilename,meanTr,stdTr,\
                  x_train,x_train_missing,mask_train,gt_train,sat_train,time_train,\
                  x_train_OI,lday_train,model_AE,DIMCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    # ***************** #
    # model compilation #
    # ***************** #

    ## model parameters
    NbProjection   = [2,2,5,5,10,12,15]
    #NbGradIter     = [2,5,7,9,10,12,15]
    NbGradIter     = [0,0,0,0,0,0,0]
    #lrUpdate       = [1e-3,1e-4,1e-5,1e-6]
    lrUpdate       = [1e-4,1e-5,1e-6,1e-7]
    IterUpdate     = [0,3,10,15,20,25,30,35,40]
    val_split      = 0.1
    iterInit       = 0
    comptUpdate    = 0  
    ## modify/check data format
    x_train         = np.nan_to_num(np.moveaxis(x_train, -1, 1))
    x_train_missing = np.nan_to_num(np.moveaxis(x_train_missing, -1, 1))
    mask_train      = np.nan_to_num(np.moveaxis(mask_train, -1, 1))
    gt_train        = np.nan_to_num(np.moveaxis(gt_train, -1, 1))
    sat_train       = np.nan_to_num(np.moveaxis(sat_train, -1, 1))
    time_train      = np.nan_to_num(np.moveaxis(time_train, -1, 1))

    # first initialize the solution
    x_train_init    = x_train_missing

    print("... Training datashape    : "+str(x_train.shape))
    pre = preprocessing.LabelEncoder()
    pre.fit(np.asarray(['','alg','h2g','j2g','j2n','j3','s3a','c2']))
    ## define dataloaders with randomised batches (no random shuffling for validation/test data)
    training_dataset = torch.utils.data.TensorDataset(\
                            torch.Tensor(x_train_init),\
                            torch.Tensor(x_train_missing),\
                            torch.Tensor(mask_train),\
                            torch.Tensor(gt_train),\
                            torch.Tensor(pre.transform(sat_train.flatten()).reshape(sat_train.shape)),\
                            torch.Tensor(time_train))
    dataset_sizes = {'train': len(training_dataset)} 
    ## instantiate model for GPU implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(".... Device GPU: "+str(torch.cuda.is_available()))
    ## initialize or load the model (bug for number of FP iterations = 0) 
    shapeData       = x_train.shape[1:]  
    NBProjCurrent = NbProjection[0]
    NBGradCurrent = NbGradIter[0]
    print('..... DinAE learning (initialisation): NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
    model = Model_4DVarNN_GradFP(\
              model_AE,shapeData,NBProjCurrent,NBGradCurrent,\
              flagGradModel,flagOptimMethod,N_cov=N_cov)    
    model =  model_to_MultiGPU(model)
    ## create an optimizer object (Adam with lr 1e-3)
    lrCurrent   = lrUpdate[0]
    lambda_LRAE = 0.5
    optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},\
                              {'params': model.model_AE.encoder.parameters(),\
                               'lr': lambda_LRAE*lrCurrent}
                              ], lr=lrCurrent)

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
    for iter in range(iterInit,Niter):
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

            # copy model parameters from current model
            model.load_state_dict(best_model_wts)                            
            # update comptUpdate
            if comptUpdate < len(NbProjection)-1:
                comptUpdate += 1
        ## daloader for the training phase                
        dataloaders = { 'train': torch.utils.data.DataLoader(training_dataset,\
                                 batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)}

        ## run NbEpoc training epochs
        for epoch in range(NbEpoc):
            print('Epoc %d/%d'%(epoch,NbEpoc))
            # Each epoch has only a training phasein OSE setup
            model.train()  # Set model to training mode
            print('..... Training step')
            running_loss         = 0.0
            running_loss_All     = 0.
            running_loss_R       = 0.
            running_loss_I       = 0.
            running_loss_AE      = 0.
            num_loss     = 0
            # Iterate over data.
            compt = 0
            for inputs_init,inputs_missing,masks,targets_GT,sat,Time in dataloaders['train']:
                compt = compt+1
                inputs_init    = inputs_init.to(device)
                inputs_missing = inputs_missing.to(device)
                masks          = masks.to(device)
                index = np.arange(0,masks.shape[1],N_cov+1)
                masks_GT       = masks[:,index,:,:]
                targets_GT     = targets_GT.to(device)
                sat            = sat.to(device)
                Time           = Time.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # need to evaluate grad/backward during the evaluation and training phase for model_AE
                with torch.set_grad_enabled(True), torch.autograd.set_detect_anomaly(True): 
                    inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
                    if model.OptimType == 1:
                        outputs,grad_new,normgrad = model(inputs_init,inputs_missing,masks,None)
                    elif model.OptimType == 2:
                        outputs,hidden_new,cell_new,normgrad = model(inputs_init,inputs_missing,masks,None,None)
                    else:
                        outputs,normgrad = model(inputs_init,inputs_missing,masks)
                    # compute Gradient of the outputs
                    Grad_pred = gradient_imageTS(outputs)
                    Grad_true = gradient_imageTS(targets_GT)
                    loss_Grad = torch.mean((Grad_pred-Grad_true)**2)
                    # compute losses
                    loss_R      = torch.sum((outputs - targets_GT)**2 * masks_GT )
                    loss_R      = torch.mul(1.0 / torch.sum(masks_GT),loss_R)
                    loss_I      = torch.sum((outputs - targets_GT)**2 * (1. - masks_GT) )
                    loss_I      = torch.mul(1.0 / torch.sum(1.-masks_GT),loss_I)
                    loss_All    = torch.mean((outputs - targets_GT)**2 )
                    if N_cov>0:
                        outputs_wcov = add_covariates_to_tensor(outputs,inputs_missing,N_cov).to(device) 
                        targets_GT_wcov = add_covariates_to_tensor(targets_GT,inputs_missing,N_cov).to(device)
                    else:
                        outputs_wcov = outputs
                        targets_GT_wcov = targets_GT
                    loss_AE     = torch.mean((model.model_AE(outputs_wcov) - outputs)**2 )
                    loss_AE_GT  = torch.mean((model.model_AE(targets_GT_wcov) - targets_GT)**2 )
                    index      = np.arange(0,inputs_missing.shape[1],N_cov+1)
                    loss_Obs    = torch.sum( (outputs - inputs_missing[:,index,:,:])**2 * masks_GT )
                    loss_Obs    = loss_Obs / torch.sum( masks_GT )

                    if flagTrWMissingData == 2:
                        loss        = alpha4DVar[0] * loss_Obs + alpha4DVar[1] * loss_AE
                        loss        = torch.add(loss_R,torch.mul(1.0,loss_AE))
                        loss        = loss_R
                        if wregul==True:
                            #print('loss_R={:4f}'.format(loss))
                            #loss_alt_Grad = torch.tensor(0., requires_grad=True)
                            loss_alt_Grad = along_track_gradient_loss(
                                                          inputs_missing[:,index[6],:,:].cpu().detach(),
                                                          masks_GT[:,6,:,:].cpu().detach(),
                                                          outputs[:,6,:,:].cpu().detach(),
                                                          Time[:,6,:,:].cpu().detach(),
                                                          sat[:,6,:,:].cpu().detach())
                            #print('loss_alt_Grad={:4f}'.format(loss_alt_Grad))
                            loss    = torch.add(regul[0]*loss,\
                                                regul[1]*loss_alt_Grad)
                            # Grad regularization
                            '''loss_Grad = torch.mean(Grad_pred**2)
                            print('loss_Grad={:4f}'.format(loss_Grad))
                            loss    = torch.add(regul[0]*loss,\
                                                regul[1]*loss_Grad)'''
                            # Lp weight regularization
                            '''Lp_reg = torch.tensor(0., requires_grad=True)
                            Lp_reg = Lp_reg.to(device)
                            for name, param in model.model_AE.encoder.named_parameters():
                                if 'weight' in name:
                                    Lp_reg = Lp_reg + torch.norm(param, 2)
                            for name, param in model.model_Grad.named_parameters():
                                if 'weight' in name:
                                    Lp_reg = Lp_reg + torch.norm(param, 2)
                            loss = loss + 10e-3 * Lp_reg'''
                    else:
                        loss        = alpha_Grad * loss_All + 0.5 * alpha_AE * ( loss_AE + loss_AE_GT )
                        #loss        = loss_All + torch.mean(Grad_pred**2) 

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
    
                # statistics
                running_loss             += loss.item() * inputs_missing.size(0)
                running_loss_I           += loss_I.item() * inputs_missing.size(0)
                running_loss_R           += loss_R.item() * inputs_missing.size(0)
                running_loss_All         += loss_All.item() * inputs_missing.size(0)
                running_loss_AE          += loss_AE_GT.item() * inputs_missing.size(0)
                num_loss                 += inputs_missing.size(0)
    
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
            if epoch_loss < best_loss:
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

        ## AE performance on training and validation datasets
        # outputs for training data
        x_train_pred = []
        for inputs_init,inputs_missing,masks,targets_GT,sat,Time in dataloaders['train']:
            inputs_init    = inputs_init.to(device)
            inputs_missing = inputs_missing.to(device)
            masks          = masks.to(device)
            targets_GT     = targets_GT.to(device)
            with torch.set_grad_enabled(True): 
                outputs_ = model(inputs_init,inputs_missing,masks)[0]
            if len(x_train_pred) == 0:
                x_train_pred  = torch.mul(1.0,outputs_).cpu().detach()
            else:
                x_train_pred  = np.concatenate((x_train_pred,\
                                   torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

        ## AE performance of the trained AE applied to gap-free data
        # ouputs for training data
        rec_AE_Tr = []
        for inputs_init,inputs_missing,masks,targets_GT,sat,Time in dataloaders['train']:
            inputs_init    = inputs_init.to(device)
            inputs_missing = inputs_missing.to(device)
            masks          = masks.to(device)
            targets_GT      = targets_GT.to(device)
            if N_cov>0:
                targets_GT_wcov = add_covariates_to_tensor(targets_GT,\
                                inputs_missing,N_cov).to(device)
            else:
                targets_GT_wcov = targets_GT
            with torch.set_grad_enabled(True): 
                outputs_ = model.model_AE(targets_GT_wcov)
            print(outputs_.shape)
            if len(rec_AE_Tr) == 0:
                rec_AE_Tr  = torch.mul(1.0,outputs_).cpu().detach()
            else:
                rec_AE_Tr  = np.concatenate((rec_AE_Tr,\
                               torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)
            print(rec_AE_Tr.shape)

        index = np.arange(0,x_train.shape[1],N_cov+1)
        mse_train,exp_var_train,\
        mse_train_interp,exp_var_train_interp=\
        eval_InterpPerformance(mask_train[:,index,:,:],x_train[:,index,:,:],x_train_missing[:,index,:,:],x_train_pred)
        exp_var_AE_Tr = eval_AEPerformance(x_train[:,index,:,:],rec_AE_Tr)
        
        if not isinstance(stdTr, list) :
            meanTr=[meanTr]
            stdTr=[stdTr]
        print(".......... iter %d"%(iter))
        print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr[0]**2,100.*exp_var_train[1]))
        print('....')
        print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr[0]**2,100.*exp_var_train[0]))
        print('....')
        print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr[0]**2,100.*exp_var_train_interp))

        # **************************** #
        # Plot figures and Save models #
        # **************************** #

        ## save models
        genSuffixModel=save_Models(dict_global_Params,genFilename,alpha4DVar,\
                                   NBProjCurrent,NBGradCurrent,model_AE,model,iter)

        ## generate some plots
        plot_Figs(dict_global_Params,genFilename,genSuffixModel,\
                  gt_train,x_train_missing,mask_train,x_train_pred,rec_AE_Tr,x_train_OI,meanTr[0],stdTr[0],\
                  lday_train,iter)

        ## save results in a pickle file
        save_Pickle(dict_global_Params,\
                    gt_train,x_train_missing,x_train_pred,rec_AE_Tr,x_train_OI,meanTr[0],stdTr[0],\
                    iter)
