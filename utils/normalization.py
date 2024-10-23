import albumentations as A

def normalize(config):
    return A.Compose([A.Normalize(mean=config.normalize.mean, std=config.normalize.std)], p=1) 