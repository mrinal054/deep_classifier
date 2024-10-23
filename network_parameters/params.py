
def model_params(name, config):
    """ Register model parameters here """
    
    # Create a dictionary to store model parameters
    param = dict()
    
    if name == "EnsembleEnet2SepIn":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["dropout"] = config.model.dropout
        
    elif name == "EnsembleRes18SepIn":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["dropout"] = config.model.dropout

    elif name == "ResNet50Pscse_256x28x28":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout
        param["activation"] = config.model.activation
        param["reduction"] = config.model.reduction
        
    elif name == "ResNet50Pscse_512x28x28":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout
        param["activation"] = config.model.activation
        param["reduction"] = config.model.reduction
        
    elif name == "EfficientNetB2LPscse_384x28x28":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout
        param["activation"] = config.model.activation
        param["reduction"] = config.model.reduction
        
    elif name == "pretrained_models":
        param["name"] = config.model.subname
        param["num_classes"] = config.train.n_classes
        param["in_chs"] = len(config.data.concat)
        
    else:
        raise ValueError(f"{name} is not found in supported model list")

    return param
        

    