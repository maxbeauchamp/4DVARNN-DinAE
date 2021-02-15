from dinae_4dvarnn import *

from ResNetConv2d       import ResNetConv2d

def ConvAE(dict_global_Params,genFilename,shapeData):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.conv1 = torch.nn.Conv2d(shapeData[0],DimAE,(3,3),padding=int(3/2))
            self.pool1 = torch.nn.AvgPool2d((2,2))
            self.conv2 = torch.nn.Conv2d(DimAE,2*DimAE,(3,3),padding=int(3/2))
            self.pool2 = torch.nn.AvgPool2d((2,2))
            self.conv3 = torch.nn.Conv2d(2*DimAE,4*DimAE,(3,3),padding=int(3/2))
            self.pool3 = torch.nn.AvgPool2d((2,2))
            self.conv4 = torch.nn.Conv2d(4*DimAE,8*DimAE,(3,3),padding=int(3/2))
            #self.pool4 = torch.nn.AvgPool2d((2,2))
            #self.conv5 = torch.nn.Conv2d(2*DimAE,2*DimAE,(3,3),padding=1)
            self.pool5 = torch.nn.AvgPool2d((5,5))
            self.conv6 = torch.nn.Conv2d(8*DimAE,DimAE,(1,1),padding=int(1/2))
            
        def forward(self, x):
            x = self.conv1( x )
            x = self.pool1( F.dropout(x) )
            x = self.conv2( F.relu(x) )
            x = self.pool2( F.dropout(x) )
            x = self.conv3( F.relu(x) )
            x = self.pool3( F.dropout(x) )
            x = self.conv4( F.relu(x) )
            #x = self.pool4(x)
            #x = self.conv5( F.relu(x) )
            x = self.pool5( F.dropout(x) )
            x = self.conv6( x )
            return x

    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            if domain=="OSMOSIS":
                self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,256,(16,21),stride=(25,25),bias=False,padding=int(16/2))
            else:
                self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,256,(20,20),stride=(25,25),bias=False,padding=int(20/2))
            self.conv2Tr = torch.nn.ConvTranspose2d(256,80,(2,2),stride=(2,2),bias=False,padding=(0,0))
            self.conv3   = torch.nn.Conv2d(80,40,(3,3),stride=(1,1),padding=int(3/2))
            self.resnet  = ResNetConv2d(2,40,2,3,int(3/2))
            self.convF  = torch.nn.Conv2d(40,int(shapeData[0]/(N_cov+1)),(1,1),padding=int(1/2))
            
        def forward(self, x):
            x = self.conv1Tr( x )
            x = self.conv2Tr( F.dropout(x) )
            x = self.conv3( F.dropout(x) )
            x = self.resnet(x)
            x = self.convF(x)
            return x

    return Encoder, Decoder
