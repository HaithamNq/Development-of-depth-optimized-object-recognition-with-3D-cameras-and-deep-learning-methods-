# Import libraries and define the model and checkpoint outside the function
import sys
import os
sys.path.append(r"C:\Users\haith\PycharmProjects\automated disassembly\BB_to_Masks_Converter\ml-depth-pro\src")
import numpy as np
from PIL import Image
import torch
import depth_pro
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------------------------------------------
# Path to the model checkpoint:

CHECKPOINT_PATH = r"C:\Users\haith\PycharmProjects\automated disassembly\BB_to_Masks_Converter\checkpoints\depth_pro.pt"
#-------------------------------------------------------------------------------------------------------------------
# Initialize the model and preprocessing transform globally:

model, transform = depth_pro.create_model_and_transforms()
model.eval()  

#-------------------------------------------------------------------------------------------------------------------
# a function to generate RGB and depth data:

def generate_rgb_and_depth(image_path):
    """
    Converts an image path to an RGB NumPy array and generates a depth map using Depth Pro.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        tuple: A tuple containing:
            - RGB_np (numpy.ndarray): The RGB image as a NumPy array (height, width, 3).
            - depth_np (numpy.ndarray): The depth map as a NumPy array (height, width).
    """
   
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)  
    prediction = model.infer(image, f_px=f_px)
    depth_np = prediction["depth"].squeeze().cpu().numpy()  
    original_image = Image.open(image_path).convert("RGB")
    RGB_np = np.array(original_image)  

    return RGB_np, depth_np


def get_camera_parameters(image_path):
    """
    Returns the camera parameters including focal lengths and principal points using Depth Pro.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        tuple: A tuple containing:
            - cx (float): The x-coordinate of the principal point.
            - cy (float): The y-coordinate of the principal point.
            - fx (float): The focal length in the x direction.
            - fy (float): The focal length in the y direction.
    """
    # Load the RGB image using Depth Pro's function
    image, _, f_px = depth_pro.load_rgb(image_path)
    
    if f_px is None:
        f_px = 1.93  # in mm

    # Get image dimensions
    height, width = image.shape[:2]
    print(f"f_px: {f_px}")
    # Assuming fx and fy are the same and f_px is the focal length in pixels

    fx = f_px
    fy = f_px
    
    # Principal point (cx, cy) is typically at the center of the image
    cx = width / 2
    cy = height / 2
    
    return cx, cy, fx, fy




def assign_depth_to_masks(masks_data, depth_array):

    """

    Assign depth values to pixels in masks based on a depth array.
    Parameters:

        mask_data (dict): A dictionary where keys are labels (e.g., "screw_10") and values are lists of (x, y) tuples.

        depth_array (numpy.ndarray): A 2D or 3D NumPy array containing depth values.
    Returns:

        dict: A dictionary with the same keys as mask_data, but values are lists of corresponding depth values.

    """

    depth_data = {}
    for key, pixels in masks_data.items():
        # Initialize a list to store depth values for this key
        depth_values = []
        for (x, y) in pixels:
            # Ensure the pixel coordinates are within the depth array bounds
            if 0 <= y < depth_array.shape[0] and 0 <= x < depth_array.shape[1]:
                depth_value = depth_array[y, x]  # Extract depth value at pixel (x, y)
                depth_values.append(float(depth_value))
            else:
                # Handle case where pixel is out of bounds (optional: assign None or a default value)
                depth_values.append(None)
        # Add the depth values list to the result dictionary under the same key
        depth_data[key] = depth_values

    return depth_data




def calculate_real_world_coordinates(screws_masks_data, screws_depth_data, cx, cy, fx, fy):
    """
    Calculates real-world coordinates (X, Y, Z) for each screw mask and appends the results to the dictionary_of_objects.

    Parameters:
        screws_masks_data (dict): Dictionary with screw IDs as keys and lists of (x, y) pixel coordinates as values.
        screws_depth_data (dict): Dictionary with screw IDs as keys and lists of corresponding depth values as values.
        cx (float): Principal point x-coordinate of the camera.
        cy (float): Principal point y-coordinate of the camera.
        fx (float): Focal length in the x-direction.
        fy (float): Focal length in the y-direction.

    Returns:
        dict: Updated dictionary_of_objects with real-world coordinates for each screw.
    """
    dictionary_of_objects = {}

    for screw_id in screws_masks_data.keys():
        pixel_coords = screws_masks_data[screw_id]  # List of (x, y) pixel coordinates
        depth_values = screws_depth_data[screw_id]  # List of depth values corresponding to pixel coordinates

        # Ensure pixel_coords and depth_values are aligned
        if len(pixel_coords) != len(depth_values):
            print(f"Error: Mismatched pixel and depth data for {screw_id}. Skipping...")
            continue

        # Compute real-world coordinates
        real_world_coordinates = []
        for i, (u, v) in enumerate(pixel_coords):
            Z = depth_values[i]*100
            X = ((u - cx) * Z / fx)/1000
            Y = ((v - cy) * Z / fy)/1000
            real_world_coordinates.append((X, Y, Z))

        # Append the screw's real-world coordinates to the dictionary
        dictionary_of_objects[screw_id] = real_world_coordinates

    return dictionary_of_objects



# Combine (x, y) with z to form (x, y, z) and merge dictionaries
def combine_dictionaries_with_depth(screws_masks_data, screws_depth_data, 
                                     module_base_masks_data, module_base_depth_data):
    # Create new dictionaries to store (x, y, z) tuples
    updated_screws_data = {
        key: [(x, y, z) for (x, y), z in zip(value, screws_depth_data[key])]  # Combine (x, y) with z
        for key, value in screws_masks_data.items()
        if key in screws_depth_data  # Ensure key exists in depth data
    }

    updated_module_data = {
        key: [(x, y, z) for (x, y), z in zip(value, module_base_depth_data[key])]  # Combine (x, y) with z
        for key, value in module_base_masks_data.items()
        if key in module_base_depth_data  # Ensure key exists in depth data
    }

    # Combine the two dictionaries into one
    combined_data = {**updated_screws_data, **updated_module_data}

    return combined_data

# Function to print samples from the combined data
def print_samples_data(combined_data, num_samples=4):
    print("Samples from combined data:")
    for key, value in list(combined_data.items())[:num_samples]:  # Take first `num_samples` keys
        print(f"Key: {key}, Sample Values: {value[:num_samples]}")  # Print a few (x, y, z) samples


#Test:
def print_sample_data(combined_data, num_samples=4):
    print("Samples from combined data:")
    keys = list(combined_data.keys())
    for key in keys[:num_samples]:
        print(f"Key: {key}, Value: {combined_data[key]}")


# Visualize the depth map


#plt.imshow(RGB_np)
#plt.imshow(depth_np, cmap='jet', alpha=0.5)  # Overlay depth map on the RGB image
#plt.colorbar(label="Depth (m)")
#plt.axis('off')
#plt.show()


import numpy as np
import cv2

def filter_objects_by_depth(dictionary, image_path, depth_generation):
    """
    Filters the dictionary by removing pixels with depth values below the median depth.
    Also displays the filtered pixels on the input image.
    
    Parameters:
    dictionary (dict): Dictionary containing object names as keys and lists of (x, y) tuples as values.
    image_path (str): Path to the input image.
    depth_generation: Module or function for generating RGB and depth images.
    
    Returns:
    dict: Filtered dictionary with only pixels having depth values above or equal to the median.
    """
    
    # Generate RGB and depth images
    RGB_np, depth_np = depth_generation.generate_rgb_and_depth(image_path)
    
    # Compute median depth value
    median_depth = np.median(depth_np)
    
    # Create a copy of the dictionary to store filtered values
    filtered_dict = {}
    
    for obj, pixels in dictionary.items():
        filtered_pixels = [(x, y) for (x, y) in pixels if depth_np[y, x] >= median_depth]
        
        if filtered_pixels:
            filtered_dict[obj] = filtered_pixels
    
    # Draw the filtered pixels on the RGB image
    for obj, pixels in filtered_dict.items():
        for (x, y) in pixels:
            cv2.circle(RGB_np, (x, y), 3, (0, 255, 0), -1)  # Green dots for remaining pixels
    
    # Show the modified image
    cv2.imshow("Filtered Image", RGB_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return filtered_dict