# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:33:35 2024

@author: M324371
"""
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from nibabel.orientations import io_orientation, apply_orientation


def read_nib(dir, target_orientation):
    """
    Parameters
    ----------
    dir : Full image directory including image name with extension
    target_orientation : Define the target orientation (e.g., RAS - Right, Anterior, Superior). 
                         Example: np.array([[1, -1], [0, -1], [2, -1]])

    Returns
    -------
    reoriented_image_data 
    """
    # Read image
    img = nib.load(dir)

    # Get the raw image data as a NumPy array
    image_data = img.get_fdata()

    # Get the affine transformation (which includes orientation info)
    affine = img.affine

    # Get the current orientation of the image
    current_orientation = io_orientation(affine)

    # Define the target orientation (e.g., RAS - Right, Anterior, Superior)
    target_orientation = target_orientation # e.g. 

    # Get the transformation matrix from current orientation to target orientation
    transformation = nib.orientations.ornt_transform(current_orientation, target_orientation)

    # Apply the transformation to the image data
    reoriented_image_data = apply_orientation(image_data, transformation)
    
    return reoriented_image_data
    
    
    
if __name__ == '__main__':
    file_dir = '/research/m324371/Project/adnexal/dataset/train/BPN0152_0029_20190105_image.nii.gz'
    
    target_orientation = np.array([[1, -1], [0, -1], [2, -1]])
    
    reoriented_image_data = read_nib(file_dir, target_orientation)
    
    # Plot the reoriented image
    plt.imshow(reoriented_image_data[:, :, reoriented_image_data.shape[2] // 2], cmap='gray')
    plt.axis('off')  # Turn off axis to match ITK-Snap's clean display
    plt.show()

