from dinae_4dvarnn import *

def save_Models(dict_global_Params,genFilename,alpha_4DVar,\
                NBProjCurrent,NBGradCurrent,model_AE,model,iter,*args):   

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    genSuffixModel = 'Torch_Alpha%03d'%(100*alpha_4DVar[0]+10*alpha_4DVar[1])
    if alpha_4DVar[0] < 1.0:
        genSuffixModel = genSuffixModel+'_AETRwithTrueData'
    else:
        genSuffixModel = genSuffixModel+'_AETRwoTrueData'
                   
    genSuffixModel = genSuffixModel+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))
    genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
    genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(flagGradModel))+'_'+str('%02d'%(flagOptimMethod))+'_'+str('%02d'%(NBGradCurrent))
      
    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAE_iter%03d'%(iter)+'.mod'
    print('.................. Auto-Encoder '+fileMod)
    torch.save(model_AE.state_dict(), fileMod)

    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAEGradFP_iter%03d'%(iter)+'.mod'
    print('.................. Auto-Encoder '+fileMod)
    torch.save(model.state_dict(), fileMod)

    return genSuffixModel
