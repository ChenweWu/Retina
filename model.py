#!/usr/bin/env python

# import relevant packages
import torchvision
import torch
import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
#import gloria.gloria as gloria
import timm

# define models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inceptionv3(pretrained=True, n_classes = 2):
    model = torchvision.models.resnet50(pretrained=pretrained)
    fc_in = model.fc.in_features
    model.fc = nn.Linear(fc_in, n_classes)
    # model = model.to(DEVICE)
    return model

def resnet50(pretrained=True, n_classes = 11):
    model = torchvision.models.resnet50(pretrained=pretrained)
    fc_in = model.fc.in_features
    model.fc = nn.Linear(fc_in, n_classes)
    # model = model.to(DEVICE)
    return model

def resnet50_binary(pretrained=True, n_classes = 22):
    model = torchvision.models.resnet50(pretrained=pretrained)
    fc_in = model.fc.in_features
    model.fc = nn.Linear(fc_in, n_classes)
    # model = model.to(DEVICE)
    return model

def densenet121(pretrained=True, n_classes = 11):
    model = torchvision.models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(list(model.features)[-1].num_features, n_classes)
    # model = model.to(DEVICE)
    return model

# def efficientnet(pretrained=True, b = 0, n_classes=11):
#     assert b in range(0,8), 'invalid model type'
#     if pretrained:
#         model = EfficientNet.from_pretrained('efficientnet-b'+str(b), num_classes=n_classes)
#     else:
#         model = EfficientNet.from_name('efficientnet-b'+str(b), num_classes=n_classes)
#     model = model.to(DEVICE)
#     return model

# def gloria_model(n_classes = 11):
#     freeze = True
#     model = gloria.load_img_classification_model(num_cls=n_classes, freeze_encoder=freeze, device=DEVICE)
#     return model

class ResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 3)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output

class EfficientNetB5(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b5_ns'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, 11)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

if __name__ == "__main__":
    # unit tests if we have time/if necessary
    pass
