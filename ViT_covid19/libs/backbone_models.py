import torch


import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torch import Tensor

from torchvision import models

#Aqui estan los detalles de cada arquitectura
#https://github.com/munniomer/pytorch-tutorials/blob/master/beginner_source/finetuning_torchvision_models_tutorial.py


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class FinetuneResNet(nn.Module):
    def __init__(self, name: str, num_classes: int,pretrained: bool = True ):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        model_func = getattr(models, name)
        self.model = model_func(pretrained=pretrained)
        set_parameter_requires_grad(self.model)
        num_ftrs = self.model.fc.in_features
        _fc_layers = [nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, self.num_classes)]
        self.model.fc = nn.Sequential(*_fc_layers)


    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)  

    def feat_extractor_modules(self):
        module_list =  []
        module_list.append(self.model.conv1)
        module_list.append(self.model.bn1)
        module_list.append(self.model.layer1)
        module_list.append(self.model.layer2)
        module_list.append(self.model.layer3)
        module_list.append(self.model.layer4)
        return module_list  

    def unfreeze1_modules(self):
        module_list =  []
        module_list.append(self.model.layer3)
        module_list.append(self.model.layer4)
        return module_list  

    def unfreeze2_modules(self):
        module_list =  []
        module_list.append(self.model.conv1)
        module_list.append(self.model.bn1)
        module_list.append(self.model.layer1)
        module_list.append(self.model.layer2)
        return module_list  

class FinetuneDensenet(nn.Module):
    def __init__(self, name: str, num_classes: int,pretrained: bool = True, dropout: bool = False ):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        model_func = getattr(models, name)
        self.model = model_func(pretrained=pretrained)
        set_parameter_requires_grad(self.model)
        num_ftrs = self.model.classifier.in_features
        if dropout == False:
            _fc_layers = [nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, self.num_classes)]

        else:
            print("Using dropout in densenet")
            _fc_layers = [nn.Dropout(inplace=True), nn.Linear(num_ftrs, 256), nn.Dropout(inplace=True), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, self.num_classes)]
            
        self.model.classifier = nn.Sequential(*_fc_layers)
        
         
    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)  

    def feat_extractor_modules(self):
        module_list =  []
        module_list.append(self.model.features.conv0)
        module_list.append(self.model.features.norm0)
        module_list.append(self.model.features.denseblock1)
        module_list.append(self.model.features.transition1)
        module_list.append(self.model.features.denseblock2)
        module_list.append(self.model.features.transition2)
        module_list.append(self.model.features.denseblock3)
        module_list.append(self.model.features.transition3)
        module_list.append(self.model.features.denseblock4)
        module_list.append(self.model.features.norm5)
        return module_list  

    def unfreeze1_modules(self):
        module_list =  []
        module_list.append(self.model.features.norm5)
        module_list.append(self.model.features.denseblock4)
        module_list.append(self.model.features.transition3)
        module_list.append(self.model.features.denseblock3)
        return module_list  

    def unfreeze2_modules(self):
        module_list =  []
        module_list.append(self.model.features.conv0)
        module_list.append(self.model.features.norm0)
        module_list.append(self.model.features.denseblock1)
        module_list.append(self.model.features.transition1)
        module_list.append(self.model.features.denseblock2)
        module_list.append(self.model.features.transition2)

        return module_list  



class FinetuneResNetAttention(nn.Module):
    def __init__(self, name: str, num_classes: int,pretrained: bool = True ):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        from utils.attention_model.resnet_attention import ResidualNet

        self.model = ResidualNet("ImageNet",50,1000,'CBAM')

        set_parameter_requires_grad(self.model)
        num_ftrs = self.model.fc.in_features
        _fc_layers = [nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, self.num_classes)]
        self.model.fc = nn.Sequential(*_fc_layers)


    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)  

    def feat_extractor_modules(self):
        module_list =  []
        print(self.model.conv1)
        module_list.append(self.model.conv1)
        module_list.append(self.model.bn1)
        module_list.append(self.model.layer1)
        if self.model.bam1 is not None:
            module_list.append(self.model.bam1)
        module_list.append(self.model.layer2)
        if self.model.bam2 is not None:
            module_list.append(self.model.bam2)
        module_list.append(self.model.layer3)
        if self.model.bam3 is not None:
            module_list.append(self.model.bam3)
        module_list.append(self.model.layer4)
        return module_list  

    def unfreeze1_modules(self):
        module_list =  []
        module_list.append(self.model.layer3)
        if self.model.bam3 is not None:
            module_list.append(self.model.bam3)
        module_list.append(self.model.layer4)
        return module_list  

    def unfreeze2_modules(self):
        module_list =  []
        module_list.append(self.model.conv1)
        module_list.append(self.model.bn1)
        module_list.append(self.model.layer1)
        if self.model.bam1 is not None:
            module_list.append(self.model.bam1)
        module_list.append(self.model.layer2)
        if self.model.bam2 is not None:
            module_list.append(self.model.bam2)
        return module_list  



class ResNetTorch(nn.Module):
    
    def __init__(self,  num_classes = 2):
        super(ResNetTorch, self).__init__()
        from torchvision.models import ResNet50_Weights
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.preconv =  nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)  
        self.preconvRelu = nn.ReLU()
        with torch.no_grad():
            self.preconv.weight.fill_(0)
            self.preconv.weight[:,:,1,1] = 1
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes, bias=True)
        #self.resnet50.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(self.resnet50.fc.in_features, num_classes, bias = True))
        
    def forward(self, x):
        x = self.preconv(x)
        x = self.preconvRelu(x)
        x = self.resnet50(x)

        return x


