from 4dvarnn-dinae import *

def classifiers(dict_global_Params,y_train):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    num_classes = (np.max(y_train)+1).astype(int)
            
    class Classifier(torch.nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.fc1 = torch.nn.Linear(DimAE,32)
            self.fc2 = torch.nn.Linear(32,64)
            self.fc3 = torch.nn.Linear(64,num_classes)
            
        def forward(self, x):
            x = self.fc1( x )
            x = self.fc2( F.relu(x) )
            x = self.fc3( F.relu(x) )
            x = F.softmax( x , num_classes )
            return x

    classifier = Classifier()

    return classifier


