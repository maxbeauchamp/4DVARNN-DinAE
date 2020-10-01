from dinae_4dvarnn import *

from PIConv2d           import PIConv2d
from ResNetConv2d       import ResNetConv2d

def PINN(dict_global_Params,genFilename,shapeData):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    WFilter       = 11
    NbResUnit     = 4
    NbFilter      = 1*DimAE

    genFilename = genFilename+str('WFilter%03d_'%WFilter)+str('NFilter%03d_'%NbFilter)+str('RU%03d_'%NbResUnit)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            #self.pool1   = torch.nn.AvgPool2d((4,4))
            self.conv1   = PIConv2d(shapeData[0],bias=False)
            self.conv2   = torch.nn.Conv2d(shapeData[0],DimAE,(1,1),padding=0,bias=False)
            self.resnet1 = ResNetConv2d(NbResUnit,DimAE,5,1,0)
            #self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,DimAE,(4,4),stride=(4,4),bias=False)
            #self.resnet2 = ResNetConv2d(NbResUnit,DimAE,5,1,0)
            self.convF   = torch.nn.Conv2d(DimAE,int(shapeData[0]/(N_cov+1)),(1,1),padding=0,bias=False)

        def forward(self, x):
            #x = self.pool1( x )
            x = self.conv1(x)
            x = self.conv2( F.relu(x) )
            x = self.resnet1( x )
            #x = self.conv1Tr( x )
            #x = self.resnet2( x )
            x = self.convF( x )
            return x
            
    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            
        def forward(self, x):
            return torch.mul(1.,x)

    return Encoder, Decoder
