import torch.nn as nn


def add_dropout_rec(module, p):
    if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
        return nn.Sequential(module, nn.Dropout(p))
    for name in module._modules.keys():
        module._modules[name] = add_dropout_rec(module._modules[name], p=p)
    return module


def add_dropout(net, p=0.1, model='densenet'):
    if model in ['densenet', 'stacked']:
        for name in net.features._modules.keys():
            if name != "conv0":
                net.features._modules[name] = add_dropout_rec(net.features._modules[name], p=p)

    elif model in ['hemis', 'concat']:
        for x in ('branches', 'combined'):
            module = net._modules[x]
            for name in module._modules.keys():
                if name != "conv0":
                    module._modules[name] = add_dropout_rec(module._modules[name], p=p)

    elif model in ['multitask', 'singletask', 'dualnet']:
        for name in net.frontal_model.features._modules.keys():
            if name != "conv0":
                net.frontal_model.features._modules[name] = add_dropout_rec(net.frontal_model.features._modules[name],
                                                                            p=p)
        net.frontal_model.classifier = add_dropout_rec(net.frontal_model.classifier, p=p)

        for name in net.lateral_model.features._modules.keys():
            if name != "conv0":
                net.lateral_model.features._modules[name] = add_dropout_rec(net.lateral_model.features._modules[name],
                                                                            p=p)
        net.lateral_model.classifier = add_dropout_rec(net.lateral_model.classifier, p=p)
    else:
        print('No dropout added')
        return net
    
    net.classifier = add_dropout_rec(net.classifier, p=p)
    return net
