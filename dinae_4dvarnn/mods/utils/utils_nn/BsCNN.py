from dinae_4dvarnn import *

from ResNetConv2d       import ResNetConv2d

class BsCNN(torch.nn.Module):

    def __init__(self,DimAE,shapeData,shapeBsBasis):
        super(BsCNN, self).__init__()
        self.shapeData    = shapeData
        self.shapeBsBasis = shapeBsBasis
        self.conv1   = torch.nn.Conv2d(shapeData[0],DimAE,(3,3),padding=1,bias=False)
        self.pool1   = torch.nn.AvgPool2d((2,2))
        self.conv2   = torch.nn.Conv2d(DimAE,2*DimAE,(3,3),padding=1,bias=False)
        self.pool2   = torch.nn.AvgPool2d((2,2))
        self.conv3   = torch.nn.Conv2d(2*DimAE,2*DimAE,(3,3),padding=1,bias=False)
        self.pool3   = torch.nn.AvgPool2d((5,5))
        NbResUnit = 7
        self.resnet1 = ResNetConv2d(NbResUnit,2*DimAE,5,1,0)
        self.fc1 = torch.nn.Linear(2*DimAE*int(shapeData[1]/20)*int(shapeData[2]/20),
                                        6*shapeBsBasis[0]*shapeBsBasis[1]*shapeBsBasis[2])

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.conv1(F.relu(x))
        x = self.pool1(x)
        x = self.conv2(F.relu(x))
        x = self.pool2(x)
        x = self.conv3(F.relu(x))
        x = self.pool3(x)
        x = self.resnet1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return torch.reshape(x,(x.shape[0],6,np.prod(self.shapeBsBasis)))
