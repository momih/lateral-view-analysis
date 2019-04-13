import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, **kwargs):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, **kwargs)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def add_dropout_rec(module, p):
    if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
        return nn.Sequential(module, nn.Dropout(p))
    for name in module._modules.keys():
        module._modules[name] = add_dropout_rec(module._modules[name], p=p)

    return module

def add_dropout_hemis(net, list_modules=['branches', 'combined'], p=0.1):
    for x in list_modules:
        module = net._modules[x]
        for name in module._modules.keys():
            if name != "conv0":
                module._modules[name] = add_dropout_rec(module._modules[name], p=p)
    net.classifier = add_dropout_rec(net.classifier, p=p)
    return net

class Hemis(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), merge_at=2, 
                 n_views=2, branch_names=None, num_init_features=64, bn_size=4, 
                 drop_rate=0, num_classes=15, in_channels=1, 
                 drop_view_prob=[0.5, 0.25, 0.25], concat=False):

        super(Hemis, self).__init__()
        separate = block_config[:merge_at]
        merged = block_config[merge_at:]
        self.branches = nn.ModuleDict()
        self.drop_view_prob = drop_view_prob
        self.concat = concat
        
        for view in range(n_views):
            name = 'branch_{}'.format(view)
            layers = [('view{}_conv0'.format(view), nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                      ('view{}_norm0'.format(view), nn.BatchNorm2d(num_init_features)),
                      ('view{}_relu0'.format(view), nn.ReLU(inplace=True)),
                      ('view{}_pool0'.format(view), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]
            
            init = nn.Sequential(OrderedDict(layers))
            num_features = num_init_features

            for i, num_layers in enumerate(separate):
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                init.add_module('%s_denseblock%d' % (name, i + 1), block)
                num_features = num_features + num_layers * growth_rate
                # TODO fix this so that it is always at the last layer independent of merge_at
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                init.add_module('%s_transition%d' % (name, i + 1), trans)
                num_features = num_features // 2

            self.branches[name] = init
    
         
        # Create merge layers
        num_features = n_views * num_features # TODO better
        
        self.combined = nn.Sequential()
        for i, num_layers in enumerate(merged):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.combined.add_module('merge_denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(merged) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.combined.add_module('merge_transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.combined.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layer

        self.classifier = nn.Linear(in_features=num_features, 
                                    out_features=num_classes) 

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def _propagate_modalities(self, images):
        # TODO branch should be called by name
        features = []
        branch_names = list(self.branches.keys())
        
        for idx, xray in enumerate(images): 
            name = branch_names[idx]
            modal_features = self.branches[name](xray)
            features.append(modal_features)      
        return features        

    def forward(self, images):
        # Propagate individual views through their branches
        features = self._propagate_modalities(images)
        stacked = torch.stack(features, dim=1)
        
        # TODO adapt for only one view
        if self.training:
            # Drop view randomly
            select = np.random.choice([1, 2, 3], p=self.drop_view_prob)
            if select == 1: # Both views
                stats = torch.cat([torch.mean(stacked, dim=1), 
                                   torch.var(stacked, dim=1)], dim=1)
    
            elif select == 2: # PA only
                stats = torch.cat([features[0], torch.zeros_like(features[0])], dim=1)
            
            elif select == 3: # L only
                stats = torch.cat([torch.zeros_like(features[1]), features[1]], dim=1)
        else:
            stats = torch.cat([torch.mean(stacked, dim=1), 
                               torch.var(stacked, dim=1)], dim=1)        
           
        out = self.combined(stats)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.classifier(out)
        return out

    
class JointConcatModel(Hemis):
    def __init__(self, **kwargs):
        super(JointConcatModel, self).__init__(**kwargs)
    
    def forward(self, images):
        features = self._propagate_modalities(images)
        concatenated = torch.cat(features, dim=1)
        out = self.combined(concatenated)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.classifier(out)
        return out

        
