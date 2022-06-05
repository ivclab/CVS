
import torch, torch.nn as nn
import pretrainedmodels as ptm
import torch.nn.functional as F
import pdb
from models.normlayer import NormLayer

"""============================================================="""
class PretrainedFrozenResNet18(torch.nn.Module):
    def __init__(self,opt,frozen=True):
        super(PretrainedFrozenResNet18, self).__init__()
        self.model = ptm.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')
        if frozen:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        self.embedding = nn.Linear(512, opt.feature_size)#<--- embedding layer to control embedding size
        if opt.nsloss:
            self.fc = NormLayer(n_classes=opt.num_classes,embed_dim=opt.feature_size)
        else:
            self.fc = nn.Linear(opt.feature_size, opt.num_classes)

    def forward(self, x, feat=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        ######
        emb = self.embedding(x)
        if feat: return emb 
        ######
        out = self.fc(emb)
        return emb, out

"""============================================================="""
class PretrainedFrozenResNet34(torch.nn.Module):
    def __init__(self,opt,frozen=True):
        super(PretrainedFrozenResNet34, self).__init__()
        self.model = ptm.__dict__['resnet34'](num_classes=1000, pretrained='imagenet')
        if frozen:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        self.embedding = nn.Linear(512, opt.feature_size)#<--- embedding layer to control embedding size
        if opt.nsloss:
            self.fc = NormLayer(n_classes=opt.num_classes,embed_dim=opt.feature_size)
        else:
            self.fc = nn.Linear(opt.feature_size, opt.num_classes)
        
    def forward(self, x, feat=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        ######
        emb = self.embedding(x)
        if feat: return emb 
        ######
        out = self.fc(emb)
        return emb, out

"""============================================================="""
class PretrainedFrozenResNet50(torch.nn.Module):
    def __init__(self, opt, frozen=True):
        super(PretrainedFrozenResNet50, self).__init__()
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
        if frozen:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        self.embedding = nn.Linear(2048, opt.feature_size)#<--- embedding layer to control embedding size
        if opt.nsloss:
            self.fc = NormLayer(n_classes=opt.num_classes,embed_dim=opt.feature_size)
        else:
            self.fc = nn.Linear(opt.feature_size, opt.num_classes)

    def forward(self, x, feat=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        ######
        emb = self.embedding(x)
        if feat: return emb 
        ######
        out = self.fc(emb)
        return emb, out