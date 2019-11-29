from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.densenet import DenseNet, _DenseBlock, _Transition, get_densenet_params


class HeMIS(nn.Module):
    r"""
    HeMIS implemented with DenseNet blocks
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), merge_at=2,
                 n_views=2, branch_names=None, num_init_features=64, bn_size=4,
                 drop_rate=0, num_classes=15, in_channels=1,
                 drop_view_prob=0.25, concat=False):

        super(HeMIS, self).__init__()

        self.branch_names = branch_names
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.in_channels = in_channels
        self.bn_size = bn_size
        self.num_init_features = num_init_features
        self.n_views = n_views
        self.num_classes = num_classes
        self.merge_at = merge_at

        separate = block_config[:merge_at]
        merged = block_config[merge_at:]

        self.branches = nn.ModuleDict()
        self.drop_view_prob = [1 - drop_view_prob, drop_view_prob/2., drop_view_prob/2.]
        self.concat = concat

        for view in range(n_views):
            name = 'branch_{}'.format(view)
            layers = [('view{}_conv0'.format(view),
                       nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
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
        num_features = n_views * num_features  # TODO better

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
        self.classifier_in_features = num_features
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
            if select == 1:  # Both views
                stats = torch.cat([torch.mean(stacked, dim=1),
                                   torch.var(stacked, dim=1)], dim=1)

            elif select == 2:  # PA only
                stats = torch.cat([features[0], torch.zeros_like(features[0])], dim=1)

            elif select == 3:  # L only
                stats = torch.cat([torch.zeros_like(features[1]), features[1]], dim=1)
        else:
            stats = torch.cat([torch.mean(stacked, dim=1),
                               torch.var(stacked, dim=1)], dim=1)

        out = self.combined(stats)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.classifier(out)
        return out


class HeMISConcat(HeMIS):
    def __init__(self, **kwargs):
        super(HeMISConcat, self).__init__(**kwargs)

    def forward(self, images):
        features = self._propagate_modalities(images)
        concatenated = torch.cat(features, dim=1)
        out = self.combined(concatenated)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.classifier(out)
        return out


class MultiViewCNN(nn.Module):
    def __init__(self, num_classes=10, combine_at='prepool', join_how='concat', multitask=True,
                 drop_view_prob=0.0, architecture='densenet121'):
        super(MultiViewCNN, self).__init__()

        self.multitask = multitask
        self.drop_view_prob = [1 - drop_view_prob, drop_view_prob/2., drop_view_prob/2.]
        if multitask:
            # Never drop view when multitask
            # Use curriculum learning on loss instead
            self.drop_view_prob = [1., 0., 0.]

        self.combine_at = combine_at
        self.join_how = join_how

        params = {'in_channels': 1, 'num_classes': num_classes, **get_densenet_params(architecture)}
        self.frontal_model = DenseNet(**params)
        self.lateral_model = DenseNet(**params)
        self.joint_in_features = self.frontal_model.classifier.in_features

        if join_how == 'concat':
            self.joint_in_features *= 2
        self.classifier = nn.Linear(in_features=self.joint_in_features, out_features=num_classes)

    def _combine_tensors(self, list_of_features, random_drop=1):
        if self.join_how == 'mean' and random_drop == 1: # average
            combined = torch.mean(torch.stack(list_of_features, dim=1), dim=1)
        elif self.join_how == 'max' and random_drop == 1:
            combined = torch.max(torch.stack(list_of_features, dim=1), dim=1)[0]
        else:  # average
            combined = torch.cat(list_of_features, dim=1)
        return combined

    def _pool(self, features):
        x = F.relu(features)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.view(features.size(0), -1)
        return x

    def forward(self, images):
        # Randomly drop a view while training
        select = np.random.choice([1, 2, 3], p=self.drop_view_prob)
        if select == 2 and self.training: # Frontal only
            frontal_img = images[0]
            lateral_img = torch.zeros_like(images[1])
        elif select == 3 and self.training: # Lateral only
            frontal_img = torch.zeros_like(images[0])
            lateral_img = images[1]
        else: # Keep both views
            frontal_img, lateral_img = images

        frontal_features = self.frontal_model.features(frontal_img)
        lateral_features = self.lateral_model.features(lateral_img)

        # Joint view
        if self.combine_at == 'prepool':
            joint = self._combine_tensors([frontal_features, lateral_features], random_drop=select)
            joint = self._pool(joint)
        else:
            # Combine after pooling
            pooled = []
            for view in [frontal_features, lateral_features]:
                pooled.append(self._pool(view))
            joint = self._combine_tensors(pooled)

        joint_logit = self.classifier(joint)

        if self.multitask:
            pooled_frontal_features = self._pool(frontal_features)
            frontal_logit = self.frontal_model.classifier(pooled_frontal_features)

            pooled_lateral_features = self._pool(lateral_features)
            lateral_logit = self.lateral_model.classifier(pooled_lateral_features)
            return joint_logit, frontal_logit, lateral_logit
        else:
            return joint_logit