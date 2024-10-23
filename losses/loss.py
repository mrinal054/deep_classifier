import torch.nn as nn

def loss_func(name, *args):
    if name == 'ce':
        return nn.CrossEntropyLoss()
    elif name == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"{name} is not found in supported losses.")