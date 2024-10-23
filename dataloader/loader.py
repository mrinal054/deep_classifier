#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import albumentations as A
from preprocessing_v2 import preprocess
import nibabel as nib


# In[3]:


class MyDataset(Dataset):
    def __init__(self, 
                 dataframe, 
                 n_classes:int=None, 
                 transform=None, 
                 normalize=None,
                 one_hot:bool=None,
                preprocess_dict:dict=None,
                concat:list=None):
        
        self.dataframe = dataframe
        self.n_classes = n_classes
        self.transform = transform
        self.normalize = normalize
        self.one_hot = one_hot 
        self.preprocess_dict = preprocess_dict
        self.concat = concat
        self.preprocess_dict = preprocess_dict
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        name = self.dataframe["Base names"][index] # image location
        
        # Do preprocessing
        pp_out_dict = preprocess(name=name, **self.preprocess_dict)

        image = pp_out_dict["image"]
        mask = pp_out_dict["mask"]
        adnexal = pp_out_dict["adnexal"]
        fluid = pp_out_dict["fluid"]
        solid = pp_out_dict["solid"]

        # Create multi-channel image if self.concat, otherwise make a copy of 'image'
        if self.concat is not None:
            list_imgs = []
            for key in self.concat: # list images defined by self.concat
                if key in pp_out_dict: list_imgs.append(pp_out_dict[key])
                else: raise ValueError(f"Key '{key}' not found in the dictionary. Possible keys are: image, adnexal, fluid, solid, and mask.")
            
            im = np.stack(list_imgs, axis=-1) 
        else: im = image.copy()

        # Augmentation
        if self.transform: im = self.transform(image=im)['image']

        # Normalize
        if self.normalize: im = self.normalize(image=im)['image']

        # Convert to tensor (always) 
        im = torch.from_numpy(im).permute(2,0,1).float() # change HWC to CHW

        # Get the class value
        y = self.dataframe["Class"][index] # classes
        
        # One-hot
        if self.one_hot: y = torch.tensor(np.eye(self.n_classes)[y], dtype=torch.float32)  # One-hot encoded tensor
        else: y = torch.tensor(y, dtype=torch.float).unsqueeze(0)   # Scalar label as a tensor
       
        return im, y, name # tensors


# In[4]:


# =============================================================================
#Test Run
# Check dataset class
if __name__ == '__main__':
    #%% Parameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 32
    ONE_HOT = True
    N_CLASSES = 2
    ONLY_ADNEXAL = False
    ONLY_FLUID = True
    ONLY_SOLID = True
    DRAW_BBOX = False
    CROP_ROI = True
    MARGIN = 200
    RESIZE = True
    KEEP_ASPECT_RATIO = True
    TARGET_SIZE = (256,256)
    CONCAT = ["image", "solid", "fluid"] # Possible keywords are "image", "adnexal", "fluid", "solid", "mask"
    
    # Read dataframe
    df_file = '/research/m324371/Project/adnexal/adnexal_dataset_all.xlsx'
    
    df = pd.read_excel(df_file, sheet_name=None) 
    
    # Read train and test sheets
    train_df = df['train']  # dataframe has a column for image names and another 
                            # column for class values
    
    train_im_dir = '/research/m324371/Project/adnexal/dataset/train'
    
    # Example normalization for grayscale images
    normalize_transform = A.Compose([A.Normalize(mean=(0,), std=(1,))]) # Example normalization for grayscale images
    # normalize_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))], p=1) # always normalize
    
    # Define the transformation pipeline
    transform_list = [
        A.HorizontalFlip(p=0.5),
        
        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=0, shift_limit=0, p=0.5, border_mode=0), # scale only
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=10, shift_limit=0, p=0.5, border_mode=0), # rotate only
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0), # shift only
                A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0), # affine transform
            ], p=0.7
        ),
           
        A.ElasticTransform(alpha=3.0, sigma=50.0, alpha_affine=None, p=0.5),
    ]
    
    train_transform = A.Compose(transform_list, p=0.0) # do augmentation 90% time
    
    train_pp_dict = {}
    train_pp_dict["file_dir"] = train_im_dir
    train_pp_dict["only_adnexal"] = ONLY_ADNEXAL
    train_pp_dict["only_fluid"] = ONLY_FLUID
    train_pp_dict["only_solid"] = ONLY_SOLID
    train_pp_dict["draw_bbox"] = DRAW_BBOX
    train_pp_dict["crop_roi"] = CROP_ROI
    train_pp_dict["margin"] = MARGIN
    train_pp_dict["resize"] = RESIZE
    train_pp_dict["keep_aspect_ratio"] = KEEP_ASPECT_RATIO
    train_pp_dict["target_size"] = TARGET_SIZE
    
    train_dataset = MyDataset(
        train_df, 
        n_classes=N_CLASSES, 
        transform=train_transform, 
        normalize=normalize_transform,
        one_hot=ONE_HOT,
        preprocess_dict=train_pp_dict,
        concat=CONCAT,
        )
    
    itr = iter(train_dataset)
    x,y,name = next(itr)
    print(x.shape, y.shape)
    x = x.numpy()
    x = np.transpose(x, [1,2,0])
    x0, x1, x2 = x[:,:,0], x[:,:,1], x[:,:,2]
    
    import matplotlib.pyplot as plt
    plt.figure(), plt.imshow(x)
    plt.figure(), plt.imshow(x0, cmap='gray')
    plt.figure(), plt.imshow(x1, cmap='gray')
    plt.figure(), plt.imshow(x2, cmap='gray')
    
    print(x.min(), x.max())
    print(x0.min(), x0.max())
    print(x1.min(), x1.max())
    print(x2.min(), x2.max())
    
    # Check dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1) 
    itr2 = iter(train_loader)
    x, y, name = next(itr2)
    
    print(x.shape)
    print(y.shape)


# In[ ]:




