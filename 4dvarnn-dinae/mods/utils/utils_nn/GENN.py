from 4dvarnn-dinae import *
sys.path.insert(0,'/linkhome/rech/genimt01/uba22to/4DVARNN-DinAE/4dvarnn-dinae/mods')

from .mods.utils.utils_nn.ConstrainedConv1d  import ConstrainedConv1d
from .mods.utils.utils_nn.ConstrainedConv2d  import ConstrainedConv2d
from .mods.utils.utils_nn.ResNetConv2d       import ResNetConv2d

def GENN(dict_global_Params,genFilename,shapeData):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    WFilter       = 5#11#
    NbResUnit     = 3#10#
    NbFilter      = 5*DimAE#20*DimAE

    genFilename = genFilename+str('WFilter%03d_'%WFilter)+str('NFilter%03d_'%NbFilter)+str('RU%03d_'%NbResUnit)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.pool1   = torch.nn.AvgPool2d((4,4))
            self.conv1   = ConstrainedConv2d(shapeData[0],NbFilter,(WFilter,WFilter),padding=int(WFilter/2),bias=False)                      
            self.conv2   = torch.nn.Conv2d(NbFilter,DimAE,(1,1),padding=0,bias=False)
            self.resnet1 = ResNetConv2d(NbResUnit,DimAE,5,1,0)
            self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,DimAE,(4,4),stride=(4,4),bias=False)
            self.resnet2 = ResNetConv2d(NbResUnit,DimAE,5,1,0)
            self.convF   = torch.nn.Conv2d(DimAE,shapeData[0],(1,1),padding=0,bias=False)
                
        def _make_ResNet(self,Nblocks,dim,K,kernel_size, padding):
            layers = []
            for kk in range(0,Nblocks):
                layers.append(torch.nn.Conv2d(dim,K*dim,kernel_size,padding=padding,bias=False))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Conv2d(K*dim,dim,kernel_size,padding=padding,bias=False))
            return torch.nn.Sequential(*layers)

        def forward(self, x):
            x = self.pool1( x )
            x = self.conv1(x)
            x = self.conv2( F.relu(x) )
            x = self.resnet1( x )
            x = self.conv1Tr( x )
            x = self.resnet2( x )
            x = self.convF( x )
            return x
            
    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            
        def forward(self, x):
            return torch.mul(1.,x)

    return Encoder, Decoder
