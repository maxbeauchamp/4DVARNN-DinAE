from 4dvarnn-dinae import *
sys.path.insert(0,'/linkhome/rech/genimt01/uba22to/4DVARNN-DinAE/4dvarnn-dinae/mods')

from .mods.utils.utils_nn.ResNetConv2d       import ResNetConv2d

def ConvAE(dict_global_Params,genFilename,shapeData):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.conv1 = torch.nn.Conv2d(shapeData[0],DimAE,(3,3),padding=1)
            self.pool1 = torch.nn.AvgPool2d((2,2))
            self.conv2 = torch.nn.Conv2d(DimAE,2*DimAE,(3,3),padding=1)
            self.pool2 = torch.nn.AvgPool2d((2,2))
            self.conv3 = torch.nn.Conv2d(2*DimAE,2*DimAE,(3,3),padding=1)
            self.pool3 = torch.nn.AvgPool2d((2,2))
            self.conv4 = torch.nn.Conv2d(2*DimAE,2*DimAE,(3,3),padding=1)
            self.pool4 = torch.nn.AvgPool2d((2,2))
            self.conv5 = torch.nn.Conv2d(2*DimAE,2*DimAE,(3,3),padding=1)
            self.pool5 = torch.nn.AvgPool2d((2,2))
            self.conv6 = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0)
            
        def forward(self, x):
            x = self.conv1( x )
            x = self.pool1(x)
            x = self.conv2( F.relu(x) )
            x = self.pool2(x)
            x = self.conv3( F.relu(x) )
            x = self.pool3(x)
            x = self.conv4( F.relu(x) )
            x = self.pool4(x)
            x = self.conv5( F.relu(x) )
            x = self.pool5(x)
            x = self.conv6( x )
            return x

    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,256,(16,16),stride=(16,16),bias=False)
            self.conv2Tr = torch.nn.ConvTranspose2d(256,64,(2,2),stride=(2,2),bias=False)
            self.conv3   = torch.nn.Conv2d(64,32,(1,1),padding=0)
            self.resnet  = ResNetConv2d(2,32,2,3,1)
            self.convF  = torch.nn.Conv2d(32,shapeData[0],(1,1),padding=0)
            
        def _make_ResNet(self,Nblocks,dim,K,kernel_size, padding):
            layers = []
            for kk in range(0,Nblocks):
                layers.append(torch.nn.Conv2d(dim,K*dim,kernel_size,padding=padding,bias=False))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Conv2d(K*dim,dim,kernel_size,padding=padding,bias=False))            
            return torch.nn.Sequential(*layers)
            
        def forward(self, x):
            x = self.conv1Tr( x )
            x = self.conv2Tr( F.relu(x) )
            x = self.conv3( F.relu(x) )
            x = self.resnet(x)
            x = self.convF(x)
            return x

    return Encoder, Decoder
