# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:43:56 2024

@author: M324371
"""
import cv2
import os
import numpy as np
import pandas as pd
from read_ultrasound import read_nib

#%%
def resize_and_pad(image, target_size=(256, 256), interpolation=cv2.INTER_CUBIC, pad_color=0):
    """
    Resize an image to fit within a target_size while maintaining the aspect ratio,
    and pad it to the exact target size with the specified pad_color (default is black).
    
    Args:
    - image (numpy array): Input image to resize and pad.
    - target_size (tuple): Target size of the frame (height, width), default is (256, 256).
    - pad_color (int or tuple): Color for padding, default is black (0). For color images, use a tuple like (0, 0, 0).
    
    Returns:
    - padded_image (numpy array): The resized and padded image.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate the scaling factor to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image while keeping the aspect ratio
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Calculate padding values to center the resized image
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    # Apply padding
    if len(image.shape) == 3:  # Color image
        padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=pad_color)
    else:  # Grayscale image
        padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=pad_color)

    return padded_image


def preprocess(file_dir:str, 
               name:str,  # base names
               only_adnexal:bool=False, 
               only_fluid:bool=False, 
               only_solid:bool=False,               
               draw_bbox:bool=False, 
               crop_roi:bool=False, 
               margin:int=0, 
               resize:bool=False, 
               keep_aspect_ratio:bool=False,
               target_size:tuple=None) -> dict:
    
    """ Peforms - bounding boxes, crop, and padding
    Args:
    - file_dir: Directory of the image location
    - name: Base name
    - only_adnexal: Whether to keep adnexal masses only. The remaining part will be removed.
    - only_fluid: Whether to keep fluid component only. The remaining part will be removed.
    - only_solid: Whether to keep solid component only. The remaining part will be removed.    
    - draw_bbox: Boolean, whether to create bounding box or not
    - crop_roi: Boolean, whether to crop ROI
    - margin: Extra region to add beyond ROI
    - resize: Boolean, whether to resize to a target size
    - keep_aspect_ratio: Whether to preserve aspect ratio
    - target_size: Needed for padding, it is the dimension of the new square-size image (e.g. (250, 250))
    
    Returns:
    - out: A dictionary
    
    """
              
    img_name = name + "_image.nii.gz"
    mask_name = name + "_mask.nii.gz"
    fs_name = name + "_fluid_QC.nii.gz" # fluid/solid name
    
    # Read image
    data = read_nib(os.path.join(file_dir, img_name), target_orientation = np.array([[1, -1], [0, -1], [2, -1]]))
    data = data.squeeze().astype('uint8')
    
    # Read mask
    mask = read_nib(os.path.join(file_dir, mask_name), target_orientation = np.array([[1, -1], [0, -1], [2, -1]]))
    mask = mask.squeeze().astype('uint8')
    mask = mask * 255 # convert from [0,1] to [0,255]
    
    # Read fluid/solid image
    fs = read_nib(os.path.join(file_dir, fs_name), target_orientation = np.array([[1, -1], [0, -1], [2, -1]]))
    fs = fs.squeeze().astype('uint8')
            
    # Get the image dimensions
    img_height, img_width = data.shape[:2]
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        print('No contour found')
           
    # Loop through each contour and get the bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)  # x, y are top-left corner, w is width, h is height
    
    # Extract fluid component
    if only_fluid:
        fluid_mask = np.where(fs == 1, 255, 0).astype(np.uint8) 
        fluid_img = cv2.bitwise_and(data, data, mask=fluid_mask) # apply the fluid mask to the original image
    else: None
    
    # extract solid component
    if only_solid:
        solid_mask = np.where(fs == 2, 255, 0).astype(np.uint8)        
        solid_img = cv2.bitwise_and(data, data, mask=solid_mask)
    else: None

    # Draw bounding boxes
    if draw_bbox:
        # Optionally, draw the bounding box on the image (for visualization)
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing
        cv2.rectangle(data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imwrite(os.path.join(save_bbox_dir, name[0] +  '.png'), data)
    
    # Extract ROI
    if crop_roi:
        # Calculate the extended bounding box with margin
        x_new = max(x - margin, 0)  # Ensure x doesn't go outside the left boundary
        y_new = max(y - margin, 0)  # Ensure y doesn't go outside the top boundary
        w_new = min(x + w + margin, img_width) - x_new  # Ensure width doesn't exceed image width
        h_new = min(y + h + margin, img_height) - y_new  # Ensure height doesn't exceed image height

        # Crop the extended bounding box from the image
        data = data[y_new:y_new+h_new, x_new:x_new+w_new]
        mask = mask[y_new:y_new+h_new, x_new:x_new+w_new]
        if only_fluid: fluid_img = fluid_img[y_new:y_new+h_new, x_new:x_new+w_new]
        if only_solid: solid_img = solid_img[y_new:y_new+h_new, x_new:x_new+w_new]
    
    # Resize preserving aspect ratios and do padding to match the target size
    if resize:
        if keep_aspect_ratio:
            data = resize_and_pad(data, target_size=target_size, interpolation=cv2.INTER_CUBIC, pad_color=0) # cubic interpolation for image
            mask = resize_and_pad(mask, target_size=target_size, interpolation=cv2.INTER_NEAREST, pad_color=0) # nearest neighbor interpolation for mask
            if only_fluid: fluid_img = resize_and_pad(fluid_img, target_size=target_size, interpolation=cv2.INTER_CUBIC, pad_color=0)
            if only_solid: solid_img = resize_and_pad(solid_img, target_size=target_size, interpolation=cv2.INTER_CUBIC, pad_color=0)
            
        else:
            data = cv2.resize(data, target_size, interpolation=cv2.INTER_CUBIC) # cubic interpolation for image
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST) # nearest neighbor interpolation for mask
            if only_fluid: fluid_img = cv2.resize(fluid_img, target_size, interpolation=cv2.INTER_CUBIC)
            if only_solid: solid_img = cv2.resize(solid_img, target_size, interpolation=cv2.INTER_CUBIC)
        
    # Keep only adnexal masses
    if only_adnexal: adnexal_img = cv2.bitwise_and(data, data, mask=mask) 
    else: adnexal_img = None
    
    # Create dictionary
    out = dict()
    out["image"] = data
    out["mask"] = mask
    out["adnexal"] = adnexal_img
    out["fluid"] = fluid_img
    out["solid"] = solid_img
    
    return out


def save(dict, save_dir):
    
    # Assert that we have required keys
    required_keys = ["names", "image", "mask", "adnexal", "fluid", "solid"]
    for k in required_keys:  assert k in dict.keys(), f"Missing required key: {k}"
    
    # Read the dictionary
    base_names = dict["names"]
    imgs = dict["image"]
    masks = dict["mask"]
    adnexals = dict["adnexal"]
    fluid = dict["fluid"]
    solid = dict["solid"]
    
    # Create directory to save
    img_save_dir = os.path.join(save_dir, "image")
    mask_save_dir = os.path.join(save_dir, "mask")
    adnexal_save_dir = os.path.join(save_dir, "adnexal")
    fluid_save_dir = os.path.join(save_dir, "fluid")
    solid_save_dir = os.path.join(save_dir, "solid")
    
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    if adnexals[0] is not None: os.makedirs(adnexal_save_dir, exist_ok=True)
    if fluid[0] is not None: os.makedirs(fluid_save_dir, exist_ok=True)
    if solid[0] is not None: os.makedirs(solid_save_dir, exist_ok=True)
    
    for base_name, img, mask, adnexal, fluid, solid in zip(base_names, imgs, masks, adnexals, fluid, solid):
        # Assuming image name does not have .png extension
        cv2.imwrite(os.path.join(img_save_dir, base_name + '_image.png'), img) 
        cv2.imwrite(os.path.join(mask_save_dir, base_name + '_mask.png'), mask)
        if adnexal is not None: cv2.imwrite(os.path.join(adnexal_save_dir, base_name + '_adnexal.png'), adnexal)
        if fluid is not None: cv2.imwrite(os.path.join(fluid_save_dir, base_name + '_fluid.png'), fluid)
        if solid is not None: cv2.imwrite(os.path.join(solid_save_dir, base_name + '_solid.png'), solid)


        
# #%% Test run
# # Image directory
# train_file_dir = 'C:/MKD/MKmayo/MKprojects/MKadnexal/Clean_stratification/train'
# test_file_dir = 'C:/MKD/MKmayo/MKprojects/MKadnexal/Clean_stratification/test'

# # Read excel files
# dir_excel = r'C:\MKD\MKmayo\MKprojects\MKadnexal' 

# xl_df = pd.read_excel(os.path.join(dir_excel, 'adnexal_dataset_all.xlsx'), sheet_name=None) # it has image info with two sheets - train and test

# # Read the train sheet for both image df and mask df
# train_df = xl_df["train"]  # for image

# # Read the test sheet for both image df and mask df
# test_df = xl_df["test"] # for image

# # Read the train and test names
# names_train = train_df["Base names"]
# names_test = test_df["Base names"]

# # Save directory
# train_save_dir = 'C:/MKD/MKmayo/MKprojects/MKadnexal/Clean_stratification/train_preprocessed'
# test_save_dir = 'C:/MKD/MKmayo/MKprojects/MKadnexal/Clean_stratification/test_preprocessed'
# os.makedirs(train_save_dir, exist_ok=True)
# os.makedirs(test_save_dir, exist_ok=True)

# # Inputs
# ONLY_ADNEXAL = True
# ONLY_FLUID = True
# ONLY_SOLID = True
# DRAW_BBOX = False
# CROP_ROI = True
# MARGIN = 200
# RESIZE = True
# KEEP_ASPECT_RATIO = True
# TARGET_SIZE = (256,256)

# # Process training data
# train_dict = preprocess(train_file_dir, 
#                         names_train, 
#                         only_adnexal=ONLY_ADNEXAL, 
#                         only_fluid=ONLY_FLUID,
#                         only_solid =ONLY_SOLID,
#                         draw_bbox=DRAW_BBOX, 
#                         crop_roi=CROP_ROI, 
#                         margin=MARGIN, 
#                         resize=RESIZE, 
#                         keep_aspect_ratio=KEEP_ASPECT_RATIO,
#                         target_size=TARGET_SIZE)

# save(train_dict, train_save_dir)

# # Process test data
# test_dict = preprocess(test_file_dir, 
#                         names_test, 
#                         only_adnexal=ONLY_ADNEXAL, 
#                         only_fluid=ONLY_FLUID,
#                         only_solid =ONLY_SOLID,
#                         draw_bbox=DRAW_BBOX, 
#                         crop_roi=CROP_ROI, 
#                         margin=MARGIN, 
#                         resize=RESIZE, 
#                         keep_aspect_ratio=KEEP_ASPECT_RATIO,
#                         target_size=TARGET_SIZE)

# save(test_dict, test_save_dir)




    
    
    
    
    


