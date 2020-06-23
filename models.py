from torchvision import models
import torch.nn as nn
import torchvision
import torch
import re


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


def resnet(num_of_classes=None):
  
    resnet18 = models.resnet18(pretrained = True)

    resnet18.fc  = torch.nn.Linear(resnet18.fc.in_features, num_of_classes)
    
    return resnet18
  
  
def densenet():
    
    densenet = models.densenet121(pretrained = True)
    
    densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, len(classes))
    
    return densenet
  
def reproduce_chexnet():
  
    PATH_TO_MODEL = "saved_models/reproduce-chexnet"
  
    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    
    return model
  
def chexnet():
  
  
    nnClassCount = 14
  
    pathModel = 'chexnet/models/m-25012018-123527.pth.tar'
    
    model = DenseNet121(nnClassCount).cuda()


  
  
    modelCheckpoint = torch.load(pathModel)

    state_dict = modelCheckpoint['state_dict']
    remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        match = pattern.match(key)
        new_key = match.group(1) + match.group(2) if match else key
        new_key = new_key[7:] if remove_data_parallel else new_key
        state_dict[new_key] = state_dict[key]
        # Delete old key only if modified.
        if match or remove_data_parallel: 
            del state_dict[key]



    model.load_state_dict(modelCheckpoint['state_dict'])
  
    print("=> loaded checkpoint")
    
    num_ftrs = model.densenet121.classifier[0].in_features
    out_size = model.densenet121.classifier[0].out_features = 2

    model.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size)
            )
    
    return model 
