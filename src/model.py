import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from utils.sam import SAM
from utils.loss import SupConLoss

class patchwiseDetector(nn.Module):

    def __init__(self,  dim_in=5120, feat_dim=512):
        super(patchwiseDetector, self).__init__()
        self.net=EfficientNet.from_pretrained("efficientnet-b0",num_classes=2)
        # self.net = Simple_CNN(2) 
        self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self,x):
        feat=self.net.extract_features(x)
        # _, embedding = self.net(x)
        # feat = self.net.pool(embedding)
        feat = feat.view(feat.shape[0], -1)
        out = F.normalize(self.head(feat), dim=1)
        return out
    
    def training_step(self,x,y):
        out = self(x)
        loss = self.cel(out,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
class patchwiseDetectorwithfreq(nn.Module):

    def __init__(self,  dim_in=7680, feat_dim=128):
        super(patchwiseDetectorwithfreq, self).__init__()
        self.net=EfficientNet.from_pretrained("efficientnet-b0",num_classes=2)
        self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self,x):
        spacial_feat=self.net.extract_features(x)
        freq_artifacts = torch.abs(torch.fft.rfft2(x))
        # print("Frequency artifact shape ", freq_artifacts.shape)
        freq_feats = self.net.extract_features(freq_artifacts)
        feat_space = spacial_feat.view(spacial_feat.shape[0],-1)
        feat_freq = freq_feats.view(freq_feats.shape[0],-1)
        feat = torch.cat((feat_space, feat_freq), dim=1)
        # feat = feat.view(feat.shape[0], -1)
        out = F.normalize(self.head(feat), dim=1)
        return out
    
class deepfakesclassifier(nn.Module):

    def __init__(self,  feature_extractor,dim_in=7680):
        super(deepfakesclassifier, self).__init__()
        self.net=feature_extractor
        self.averagepool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(dim_in, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2)
            )
        self.cel=nn.CrossEntropyLoss()
        self.optimizer=torch.optim.SGD(self.parameters(),lr=0.0001, momentum=0.9)

    def forward(self,x):
        feat=self.net(x)
        # freq_artifacts = torch.abs(torch.fft.rfft2(x))
        # freq_feats = self.net.extract_features(freq_artifacts)
        # feat_space = spacial_feat.view(spacial_feat.shape[0],-1)
        # feat_freq = freq_feats.view(freq_feats.shape[0],-1)
        # feat = torch.cat((feat_space, feat_freq), dim=1)
        out = self.head(feat)
        return out
    
    def training_step(self,x,target):
        pred_cls=self(x)
        loss=self.cel(pred_cls,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return pred_cls


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        self.cel=nn.CrossEntropyLoss()
        self.optimizer=SAM(self.parameters(),torch.optim.SGD,lr=0.001, momentum=0.9)
        
        

    def forward(self,x):
        x=self.net(x)
        return x
    
    def training_step(self,x,target):
        for i in range(2):
            pred_cls=self(x)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            loss=loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.main(input)

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)

class Simple_CNN(nn.Module):
    def __init__(self, class_num, pretrain=False):
        super(Simple_CNN, self).__init__()
        nc = 3
        nf = 64
        self.main = nn.Sequential(
            dcgan_conv(nc, nf),
            vgg_layer(nf, nf),

            dcgan_conv(nf, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf * 8, nf * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(nf * 8, class_num, bias=True)
        )
        self.pretrain = pretrain

    def forward(self, input):
        embedding = self.main(input)
        feature = self.pool(embedding)
        feature = feature.view(feature.shape[0], -1)
        cls_out = self.classification_head(feature)
        if not self.pretrain:
            cls_out = F.softmax(cls_out)
        return cls_out, embedding

class SupConNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, backbone, head='mlp', dim_in=512, feat_dim=128):
        super(SupConNet, self).__init__()
        self.backbone=backbone
        if head=='linear':
            self.head=nn.Linear(dim_in, feat_dim)
        elif head=='mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        cls_out, embedding = self.backbone(x)
        feat = self.backbone.pool(embedding)
        feat = feat.view(feat.shape[0], -1)
        feat = F.normalize(self.head(feat), dim=1)
        return cls_out, feat
