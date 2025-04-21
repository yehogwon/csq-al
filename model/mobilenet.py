import torch
import torch.nn as nn
import torchvision

class MobileNetV3s(nn.Module): 
    def __init__(self, n_classes: int): 
        super(MobileNetV3s, self).__init__()
        self.model = torchvision.models.mobilenet_v3_small(num_classes=n_classes)

        self.features = self.model.features
        self.avgpool = self.model.avgpool
        self.classifier = self.model.classifier

        self.embedding_dim = self.model.classifier[0].in_features
    
    def forward(self, x): 
        x = self.features(x)
        x = self.avgpool(x)
        emb = torch.flatten(x, 1)
        out = self.classifier(emb)
        return out, emb
    
    def get_embedding_dim(self): 
        return self.embedding_dim
