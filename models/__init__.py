from .densenet import DenseNet, get_densenet_params
from .resnet import ResNet, get_resnet_params
from .joint import HeMIS, HeMISConcat, FrontalLateralMultiTask
from .dropout import add_dropout


def create_model(model_type, num_classes, target='joint', architecture='densenet', dropout=0.0, otherargs=None):
    if target == 'joint':
        if model_type in ['singletask', 'multitask', 'dualnet']:
            joint_only = model_type != 'multitask'
            model = FrontalLateralMultiTask(num_classes=num_classes, combine_at=otherargs.combine,
                                            join_how=otherargs.join, drop_view_prob=otherargs.drop_view_prob,
                                            joint_only=joint_only, architecture=architecture)
        elif model_type == 'stacked':
            modelparams = get_densenet_params(architecture)
            model = DenseNet(num_classes=num_classes, in_channels=2, **modelparams)

        else:  # Default HeMIS
            Net = HeMISConcat if model_type == 'concat' else HeMIS
            model = Net(num_classes=num_classes, in_channels=1, merge_at=otherargs.merge,
                        drop_view_prob=otherargs.drop_view_prob)

    else:
        if 'resnet' in architecture:
            modelparams = get_resnet_params(architecture)
            model = ResNet(num_classes=num_classes, in_channels=1, **modelparams)
            model_type = 'resnet'
        else:
            # Use DenseNet by default
            modelparams = get_densenet_params(architecture)
            model = DenseNet(num_classes=num_classes, in_channels=1, **modelparams)
            model_type = 'densenet'

    print('Created {} model'.format(model_type))
    # Add dropout
    if dropout:
        model = add_dropout(model, p=dropout, model=model_type)
    return model
