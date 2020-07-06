            IterUpdate     = [1,2,6,9,12,15,20]#[0,2,4,6,9,15]
            NbProjection   = [0,2,5,5,10,10,15]#[0,0,0,0,0,0]#[5,5,5,5,5]##
            NbGradIter     = [0,0,0,0,0,0,0]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
            lrUpdate       = [1e-3,1e-4,1e-4,1e-5,1e-5,1e-6,1e-6]

            if 1*1:
                IterUpdate     = [1,2,6,9,12,15,18]#[0,2,4,6,9,15]
                NbProjection   = [0,1,1,1,1,1,1]#[0,0,0,0,0,0]#[5,5,5,5,5]##
                NbGradIter     = [0,2,5,5,10,10,15]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
                lrUpdate       = [1e-4,1e-4,1e-4,1e-5,1e-5,1e-6,1e-7]

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
            print()
            print('...........................................')
            print('....   stdTr = %.3f'%stdTr)
            print('....   stdTt = %.3f'%stdTt)

            ## Define dataloaders with randomised batches     
            ## no random shuffling for validation/test data
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_missing),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_missing),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
                        
            dataset_sizes = {'train': len(training_dataset), 'val': len(test_dataset)} 

            ## instantiate model for GPU implementation
            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cuda")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
    
    
            iterInit      = 13
            comptUpdate   = 4
            NBProjCurrent = NbProjection[comptUpdate]
            NBGradCurrent = NbGradIter[comptUpdate]
            print('..... DinAE learning (initialisation): NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
            
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
            flagLoadModelAE = 0
            fileAEModelInit = './ResSSTNATL60/model_patchDataset_NATL60withMETOP_SST_128_512_011WFilter005_NFilter100_RU003_Torch_Alpha101_AETRwoTrueData07D20N00W00_Nproj01_Grad_00_01_10_modelAEGradFP_iter015.mod'
            if flagLoadModelAE == 1:
                model.load_state_dict(torch.load(fileAEModelInit))
                
            # create an optimizer object
            # Adam optimizer with learning rate 1e-3
            lrCurrent        = lrUpdate[comptUpdate]
            optimizer        = optim.Adam(model.parameters(), lr=lrCurrent)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.)
                        
            # model compilation
            # model fit
            since = time.time()

            alpha_MaskedLoss = alpha_Losss[0]
            alpha_GTLoss     = 1. - alpha_Losss[0]
            alpha_AE         = alpha_Losss[1]

            best_model_wts = copy.deepcopy(model.state_dict())
                                    
            for iter in range(iterInit,Niter):

                print()
                print('............................................................')
                print('............................................................')
                print('..... Iter  %d '%(iter))
                best_loss      = 1e10
                if iter == IterUpdate[comptUpdate]:
                    # update GradFP parameters
                    NBProjCurrent = NbProjection[comptUpdate]
                    NBGradCurrent = NbGradIter[comptUpdate]
                    lrCurrent     = lrUpdate[comptUpdate]
                    print("..... ")
                    print("..... ")
                    print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))

                    # update GradFP architectures
                    print('..... Update model architecture')
                    print("..... ")
                    model = dinAE.Model_AE_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,flagGradModel,flagOptimMethod)
                    model = model.to(device)
                    
                    # UPDATE optimizer
                    optimizer        = optim.Adam(model.parameters(), lr= lrCurrent)
                    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

                    # copy model parameters from current model
                    model.load_state_dict(best_model_wts)
                                        
                    # update comptUpdate
                    if comptUpdate < len(NbProjection)-1:
                        comptUpdate += 1

                print('..... AE Model type : %d '%(flagAEType))
                print('..... Gradient type : %d '%(flagGradModel))
                print('..... Optim type    : %d '%(flagOptimMethod))
                print('..... DinAE learning: NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
                print('..... Learning rate : %f'%lrCurrent)
                print('..... Loss          : I-Loss %.1f -- AE-Loss %.1f'%(alpha_Losss[0],alpha_Losss[1]))

                # Daloader during training phase                
                dataloaders = {
                    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                    'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                }
                
                # Run NbEpoc training epochs
                for epoch in range(NbEpoc):
                    #print('Epoch {}/{}'.format(epoch, NbEpoc - 1))
                    #print('-' * 10)
                    print('Epoc %d/%d'%(epoch,NbEpoc))
                    
                    # Each epoch has a training and validation phase
                    for phase in ['train', 'val']:
                        if phase == 'train':
                            #rint('Learning')
                            model.train()  # Set model to training mode
                            #print('..... Training step')
                        else:
                            model.eval()   # Set model to evaluate mode
                            #print('..... Test step')
            
                        running_loss         = 0.0
                        running_loss_All     = 0.
                        running_loss_R       = 0.
                        running_loss_I       = 0.
                        running_loss_AE      = 0.
                        num_loss     = 0
    
                        # Iterate over data.
                        #for inputs_ in dataloaders[phase]:
                        #    inputs = inputs_[0].to(device)
                        compt = 0
                        for inputs_missing,masks,inputs_GT in dataloaders[phase]:
                            compt = compt+1
                            #print('.. batch %d'%compt)
                            
                            inputs_missing = inputs_missing.to(device)
                            masks          = masks.to(device)
                            inputs_GT      = inputs_GT.to(device)
                            #print(inputs.size(0))
            
                            # zero the parameter gradients
                            optimizer.zero_grad()
    
                            # forward
                            # need to evaluate grad/backward during the evaluation and training phase for model_AE
                            with torch.set_grad_enabled(True): 
                            #with torch.set_grad_enabled(phase == 'train'):
                                outputs = model(inputs_missing,masks)
                                #outputs = model(inputs)
                                #loss = criterion( outputs,  inputs)
                                loss_R      = torch.sum((outputs - inputs_GT)**2 * masks )
                                loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
                                loss_I      = torch.sum((outputs - inputs_GT)**2 * (1. - masks) )
                                loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
                                loss_All    = torch.mean((outputs - inputs_GT)**2 )
                                loss_AE     = torch.mean((model.model_AE(outputs) - outputs)**2 )
                                loss_AE_GT  = torch.mean((model.model_AE(inputs_GT) - inputs_GT)**2 )
                                
                                if phase == 'train':
                                    loss = torch.mul(1.,loss_All)
                                else :
                                    loss = torch.mul(1.,loss_R)
                                #else: 
                                #    loss = torch.mul(alpha_GTLoss,loss_All)
                                loss = torch.add(loss,torch.mul(alpha_AE,loss_AE))
            
                                # backward + optimize only if in training phase
                                if 1*1: #phase == 'train':
                                    loss.backward()
                                    optimizer.step()
    
                            # statistics
                            running_loss             += loss.item() * inputs_missing.size(0)
                            running_loss_I           += loss_I.item() * inputs_missing.size(0)
                            running_loss_R           += loss_R.item() * inputs_missing.size(0)
                            running_loss_All         += loss_All.item() * inputs_missing.size(0)
                            running_loss_AE          += loss_AE_GT.item() * inputs_missing.size(0)
                            num_loss                 += inputs_missing.size(0)
                            #running_expvar += torch.sum( (outputs - inputs)**2 ) / torch.sum(
    
                        #if phase == 'train':
                        if phase == 'val':
                            exp_lr_scheduler.step()
    
                        epoch_loss       = running_loss / num_loss
                        epoch_loss_All   = running_loss_All / num_loss
                        epoch_loss_AE    = running_loss_AE / num_loss
                        epoch_loss_I     = running_loss_I / num_loss
                        epoch_loss_R     = running_loss_R / num_loss
                        #epoch_acc = running_corrects.double() / dataset_sizes[phase]
                        
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
    
                        #print('{} Loss: {:.4f} '.format(
                         #   phase, epoch_loss))
                        print('.. {} Loss: {:.4f} NLossAll: {:.4f} NLossR: {:.4f} NLossI: {:.4f} NLossAE: {:.4f}'.format(
                            phase, epoch_loss,epoch_nloss_All,epoch_nloss_R,epoch_nloss_I,epoch_nloss_AE))
            
                        # deep copy the model
                        if phase == 'val' and epoch_loss < best_loss:
                            best_loss = epoch_loss
                            best_model_wts = copy.deepcopy(model.state_dict())
    
                        #print()
                
                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    print('Best val loss: {:4f}'.format(best_loss))


                # load best model weights
                model.load_state_dict(best_model_wts)
            
                ## Performance summary for best model
                # Daloader during training phase                
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
                
                print(".......... iter %d"%(iter))
                print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
                print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
                print('....')
                print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
                print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
                print('....')
                print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
                print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))
                    
                if flagSaveModel == 1:
                    genSuffixModel = 'Torch_Alpha%03d'%(100*alpha_Losss[0]+10*alpha_Losss[1])
#                    if flagUseMaskinEncoder == 1:
#                        genSuffixModel = genSuffixModel+'_MaskInEnc'
#                        if stdMask  > 0:
#                            genSuffixModel = genSuffixModel+'_Std%03d'%(100*stdMask)
                    genSuffixModel = genSuffixModel+'_AETRwithGTTrandTt'
                   
                    genSuffixModel = genSuffixModel+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))
                    genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
                    genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(flagGradModel))+'_'+str('%02d'%(flagOptimMethod))+'_'+str('%02d'%(NBGradCurrent))
      
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAE_iter%03d'%(iter)+'.mod'
                    print('.................. Auto-Encoder '+fileMod)
                    torch.save(model_AE.state_dict(), fileMod)

                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAEGradFP_iter%03d'%(iter)+'.mod'
                    print('.................. Auto-Encoder '+fileMod)
                    torch.save(model.state_dict(), fileMod)
