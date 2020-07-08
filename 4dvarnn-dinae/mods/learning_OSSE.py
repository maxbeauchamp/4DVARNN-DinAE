from 4dvarnn-dinae import *
from .utils.tools import *
from .utils.graphics import *
from .utils.eval_Performance      import eval_AEPerformance
from .utils.eval_Performance      import eval_InterpPerformance
from .utils.plot_Figs             import plot_Figs
from .utils.save_Pickle           import save_Pickle
from .utils.save_Models           import save_Models
sys.path.insert(0,'/linkhome/rech/genimt01/uba22to/4DVARNN-DinAE/4dvarnn-dinae/mods')

from .utils.utils_solver.Model_4DVarNN_FP      import Model_4DVarNN_FP
from .utils.utils_solver.Model_4DVarNN_Grad    import Model_4DVarNN_GradFP
from .utils.utils_solver.Model_4DVarNN_GradFP  import Model_4DVarNN_GradFP

def learning_OSSE(dict_global_Params,genFilename,\
                  x_train,x_train_missing,mask_train,gt_train,x_train_OI,lday_pred,meanTr,stdTr,\
                  x_test,x_test_missing,mask_test,gt_test,x_test_OI,lday_test,model_AE,DIMCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    # ***************** #
    # model compilation #
    # ***************** #

    ## model parameters
    NbProjection   = [2,2,5,5,10,15,14]#[0,0,2,2,5,5,10,15,14]
    NbGradIter     = [0,2,5,5,10,10,15]
    lrUpdate       = [1e-4,1e-5,1e-6,1e-7]#[1e-3,1e-4,1e-5,1e-5,1e-5,1e-6,1e-6,1e-5,1e-6]
    IterUpdate     = [0,3,10,15,20,25,30,35,40]
    val_split      = 0.1
    iterInit       = 0
    comptUpdate    = 0  
    ## modify/check data format
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
    ## define dataloaders with randomised batches (no random shuffling for validation/test data)
    training_dataset = torch.utils.data.TensorDataset(\
                            torch.Tensor(x_train_missing),\
                            torch.Tensor(mask_train),\
                            torch.Tensor(gt_train)) 
    test_dataset     = torch.utils.data.TensorDataset(\
                            torch.Tensor(x_test_missing),\
                            torch.Tensor(mask_test),\
                            torch.Tensor(gt_test))       
    dataset_sizes = {'train': len(training_dataset), 'val': len(test_dataset)} 
    ## instantiate model for GPU implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(".... Device GPU: "+str(torch.cuda.is_available()))
    ## initialize or load the model (bug for number of FP iterations = 0) 
    shapeData       = x_train.shape[1:]  
    NBProjCurrent = NbProjection[0]
    NBGradCurrent = NbGradIter[0]
    print('..... DinAE learning (initialisation): NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
    if flagLoadModelAE == 1:
        model.load_state_dict(torch.load(fileAEModelInit))
    else:
        model = Model_4DVarNN_GradFP(\
              model_AE,shapeData,NBProjCurrent,NBGradCurrent,\
              flagGradModel,flagOptimMethod,N_cov=N_cov)    
    ## create an optimizer object (Adam with lr 1e-3)
    lrCurrent        = lrUpdate[0]
    optimizer        = optim.Adam(model.parameters(), lr=lrCurrent)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.)                  
    ## model compilation 
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
            model = model.to(device)        
            # update optimizer
            optimizer        = optim.Adam(model.parameters(), lr= lrCurrent)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
            # copy model parameters from current model
            model.load_state_dict(best_model_wts)                            
            # update comptUpdate
            if comptUpdate < len(NbProjection)-1:
                comptUpdate += 1
        ## daloader for the training phase                
        dataloaders = { 'train': torch.utils.data.DataLoader(training_dataset,\
                                 batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),\
                        'val':   torch.utils.data.DataLoader(test_dataset,\
                                 batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True), }
                
        ## run NbEpoc training epochs
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

        ## AE performance on training and validation datasets
        # outputs for training data
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
        # outputs for test data
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

        mse_train,exp_var_train,\
        mse_test,exp_var_test,\
        mse_train_interp,exp_var_train_interp,\
        mse_test_interp,exp_var_test_interp =\
        eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,\
                               mask_test,x_test,x_test_missing,x_test_pred)
        exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
        
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

        ## save models
        genSuffixModel=save_Models(dict_global_Params,genFilename,alpha_Losss,\
                                   NBProjCurrent,NBGradCurrent,model_AE,model,iter)
        ## generate some plots
        plot_Figs(dirSAVE,genFilename,genSuffixModel,\
                  gt_train,x_train_missing,mask_train,x_train_pred,rec_AE_Tr,x_train_OI,meanTr,stdTr,\
                  gt_test,x_test_missing,mask_test,x_test_pred,rec_AE_Tt,x_test_OI,\
                  lday_pred,lday_test,iter)
        ## save results in a pickle file
        save_Pickle(dirSAVE,\
                    gt_train,x_train_missing,x_train_pred,rec_AE_Tr,x_train_OI,meanTr,stdTr,\     
                    gt_test,x_test_missing,x_test_pred,rec_AE_Tt,x_test_OI,\
                    iter)       
