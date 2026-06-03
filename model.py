import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models

class GradientReversalLayer(torch.autograd.Function):
    # Passes input unmodified during the forward pass while caching the scaling factor alpha
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    # Negates and scales the incoming gradient by alpha during the backward pass to reverse training signals
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainClassifier(nn.Module):
    # Set up the CNN feature mapping and dense classification layer to distinguish domains (countries)
    def __init__(self, in_channels, num_domains=12):
        super(DomainClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_domains)

    # Passes features through Gradient Reversal Layer, pooling, and projects to domain classification logits
    def forward(self, x, alpha):
        x = GradientReversalLayer.apply(x, alpha)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ContextualModule(nn.Module):
    # Builds multi-scale pooling pyramids, a 1x1 bottleneck layer, and weight network for scale adaptation
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)

    # Computes spatial weight maps by comparing local features against contextual features
    def __make_weight(self,feature,scale_feature):
        weight_feature = feature - scale_feature
        return torch.sigmoid(self.weight_net(weight_feature))

    # Helper method to create a scale block consisting of spatial average pooling and 1x1 Conv
    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    # Aggregates multi-scale pooled contextual features dynamically using learned weights
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        
        multi_scales = []
        for stage in self.scales:
            if feats.device.type == 'mps':
                # stage[0] is AdaptiveAvgPool2d (no weights), stage[1] is Conv2d (has weights)
                pooled = stage[0](feats.cpu()).to(feats.device)
                stage_out = stage[1](pooled)
            else:
                stage_out = stage(feats)
            multi_scales.append(F.interpolate(input=stage_out, size=(h, w), mode='bilinear', align_corners=False))
            
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        aggregated_feats = (multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3] + 1e-8)
        overall_features = [aggregated_feats, feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle), aggregated_feats

class CANNet(nn.Module):
    # Initializes frontend (VGG16), contextual, and dilated backend convolutional blocks of CANNet
    def __init__(self, load_weights=False):
        super(CANNet, self).__init__()
        self.seen = 0
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,batch_norm=True, dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._initialize_weights()
            fsd = list(self.frontend.state_dict().items())
            msd = list(mod.state_dict().items())
            for i in range(len(fsd)):
                fsd[i][1].data[:] = msd[i][1].data[:]

    # Standard forward pass generating the 2D crowd density map estimation from the input image
    def forward(self,x):
        x = self.frontend(x)
        bottle, _ = self.context(x)
        x = self.backend(bottle)
        x = self.output_layer(x)
        return x

    # Initializes custom convolution weights normally and batch norm layers constantly
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class CANNet_GRL_Frontend(CANNet):
    # Initializes GRL variant with a Domain Classifier attached to frontend features
    def __init__(self, load_weights=False, num_domains=12):
        super(CANNet_GRL_Frontend, self).__init__(load_weights)
        self.domain_classifier = DomainClassifier(in_channels=512, num_domains=num_domains)

    # Computes crowd density and domain classification logits using frontend features via GRL
    def forward(self, x, alpha=1.0):
        feat_frontend = self.frontend(x)
        bottle, _ = self.context(feat_frontend)
        out = self.backend(bottle)
        out = self.output_layer(out)
        domain_logits = self.domain_classifier(feat_frontend, alpha)
        return out, domain_logits

class CANNet_GRL_Context(CANNet):
    # Initializes GRL variant with a Domain Classifier attached to scale-aggregated contextual features
    def __init__(self, load_weights=False, num_domains=12):
        super(CANNet_GRL_Context, self).__init__(load_weights)
        self.domain_classifier = DomainClassifier(in_channels=512, num_domains=num_domains)

    # Computes crowd density and domain classification logits using contextual features via GRL
    def forward(self, x, alpha=1.0):
        feat_frontend = self.frontend(x)
        bottle, aggregated_feats = self.context(feat_frontend)
        out = self.backend(bottle)
        out = self.output_layer(out)
        domain_logits = self.domain_classifier(aggregated_feats, alpha)
        return out, domain_logits

class CANNet_GRL_Concat(CANNet):
    # Initializes GRL variant with a Domain Classifier attached to bottleneck concatenation features
    def __init__(self, load_weights=False, num_domains=12):
        super(CANNet_GRL_Concat, self).__init__(load_weights)
        self.domain_classifier = DomainClassifier(in_channels=512, num_domains=num_domains)

    # Computes crowd density and domain classification logits using concat bottleneck features via GRL
    def forward(self, x, alpha=1.0):
        feat_frontend = self.frontend(x)
        bottle, _ = self.context(feat_frontend)
        out = self.backend(bottle)
        out = self.output_layer(out)
        domain_logits = self.domain_classifier(bottle, alpha)
        return out, domain_logits

class CANNet_CORAL_Frontend(CANNet):
    # Initializes DeepCORAL variant extracting features from the frontend layer for alignment
    def __init__(self, load_weights=False):
        super(CANNet_CORAL_Frontend, self).__init__(load_weights)
        self.gap = nn.AdaptiveAvgPool2d(1)

    # Computes crowd density and returns globally pooled frontend features for CORAL loss computation
    def forward(self, x, alpha=1.0):
        feat_frontend = self.frontend(x)
        bottle, _ = self.context(feat_frontend)
        out = self.backend(bottle)
        out = self.output_layer(out)
        features = self.gap(feat_frontend).view(feat_frontend.size(0), -1)
        return out, features

class CANNet_CORAL_Context(CANNet):
    # Initializes DeepCORAL variant extracting features from scale-aggregated context features
    def __init__(self, load_weights=False):
        super(CANNet_CORAL_Context, self).__init__(load_weights)
        self.gap = nn.AdaptiveAvgPool2d(1)

    # Computes crowd density and returns globally pooled contextual features for CORAL loss computation
    def forward(self, x, alpha=1.0):
        feat_frontend = self.frontend(x)
        bottle, aggregated_feats = self.context(feat_frontend)
        out = self.backend(bottle)
        out = self.output_layer(out)
        features = self.gap(aggregated_feats).view(aggregated_feats.size(0), -1)
        return out, features

class CANNet_CORAL_Concat(CANNet):
    # Initializes DeepCORAL variant extracting features from bottleneck concatenation features
    def __init__(self, load_weights=False):
        super(CANNet_CORAL_Concat, self).__init__(load_weights)
        self.gap = nn.AdaptiveAvgPool2d(1)

    # Computes crowd density and returns globally pooled concatenated features for CORAL loss computation
    def forward(self, x, alpha=1.0):
        feat_frontend = self.frontend(x)
        bottle, _ = self.context(feat_frontend)
        out = self.backend(bottle)
        out = self.output_layer(out)
        features = self.gap(bottle).view(bottle.size(0), -1)
        return out, features


# Factory function to build sequential Conv2D/BatchNorm2d/ReLU blocks based on configuration lists
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
