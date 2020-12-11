from dinae_4dvarnn import *

from ConstrainedConv1d  import ConstrainedConv1d
from ConstrainedConv2d  import ConstrainedConv2d
from ResNetConv2d       import ResNetConv2d

def GENN(dict_global_Params,genFilename,shapeData):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    WFilter       = 11#5
    NbResUnit     = 7#10#3
    NbFilter      = 1*DimAE#20*DimAE

    genFilename = genFilename+str('WFilter%03d_'%WFilter)+str('NFilter%03d_'%NbFilter)+str('RU%03d_'%NbResUnit)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()

            # layers for large scale
            self.pool1   = torch.nn.AvgPool2d((int(4/dwscale),int(4/dwscale)))
            self.conv1   = ConstrainedConv2d(shapeData[0],NbFilter,(WFilter,WFilter),padding=int(WFilter/2),bias=False)
            self.conv2   = torch.nn.Conv2d(NbFilter,DimAE,(1,1),padding=0,bias=False)
            self.resnet1 = ResNetConv2d(NbResUnit,DimAE,5,1,0)
            self.conv3   = torch.nn.Conv2d(DimAE,int(shapeData[0]/(N_cov+1)),(1,1),padding=0,bias=False)
            self.conv1Tr = torch.nn.ConvTranspose2d(int(shapeData[0]/(N_cov+1)),int(shapeData[0]/(N_cov+1)),\
                                                   (int(4/dwscale),int(4/dwscale)),stride=(int(4/dwscale),int(4/dwscale)),bias=False)
            self.conv4   = torch.nn.Conv2d(int(shapeData[0]/(N_cov+1)),2*DimAE,(3,3),padding=int(3/2),bias=False)

            # layers for small scale
            self.conv1b   = ConstrainedConv2d(shapeData[0],NbFilter,(WFilter,WFilter),padding=int(WFilter/2),bias=False)
            self.conv2b   = torch.nn.Conv2d(NbFilter,DimAE,(1,1),padding=0,bias=False)
            self.resnet1b = ResNetConv2d(NbResUnit,DimAE,5,1,0)
            self.conv3b   = torch.nn.Conv2d(DimAE,2*DimAE,(1,1),padding=0,bias=False)

            # final layer
            self.convF   = torch.nn.Conv2d(2*DimAE,int(shapeData[0]/(N_cov+1)),(3,3),padding=int(3/2),bias=False)
                
        def forward(self, x):
            # large scale
            xbar = self.pool1( x )
            xbar = self.conv1( xbar )
            xbar = self.conv2( F.relu(xbar) )
            xbar = self.resnet1( xbar )
            xbar = self.conv3( xbar )
            xbar = self.conv1Tr( xbar )
            xbar = self.conv4( F.relu(xbar) )

            dx = self.conv1b( x )
            dx = self.conv2b( F.relu(dx) )
            dx = self.resnet1b( dx ) 
            dx = self.conv3b( dx )
            # large sclae + small scale
            x = self.convF( xbar + dx)
            return x
            
    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            
        def forward(self, x):
            return torch.mul(1.,x)

    return Encoder, Decoder
