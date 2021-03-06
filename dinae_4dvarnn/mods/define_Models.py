from dinae_4dvarnn import *
from ConvAE import ConvAE
from GENN import GENN
#from GENN2S import GENN
from PINN import Space_Time_ADConv2d as PINN

def define_Models(dict_global_Params,genFilename,shapeData):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    DimCAE = DimAE  
    
    if flagAEType == 1:   ## Conv-AE 
      Encoder, Decoder = ConvAE(dict_global_Params,genFilename,shapeData)     
    if flagAEType == 2:   ## GENN
      Encoder, Decoder = GENN(dict_global_Params,genFilename,shapeData)
    if flagAEType == 3:   ## PINN
      shapeBsBasis=(3,5,5)
      Encoder, Decoder = PINN(dict_global_Params,genFilename,shapeData,shapeBsBasis)

    ## auto-encoder architecture
    class Model_AE(torch.nn.Module):
        def __init__(self):
            super(Model_AE, self).__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
            
        def forward(self, x):
            x = self.encoder( x )
            x = self.decoder( x )
            return x
              
    model_AE = Model_AE()
    print(model_AE)
    print('Number of trainable parameters = %d'%(sum(p.numel() for p in model_AE.parameters() if p.requires_grad)))  

    return genFilename, Encoder, Decoder, model_AE, DimCAE
