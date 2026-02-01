import cv2 
import os
import sys 
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yolo_detectors
import depth_generation
import Topological_sorting
import objects_pipeline
import sqlite3
# Define the objects based on their functions

# Onsert different paths for the image, model, and output image:
# first difference between the 3D camera and the 2D image functions

image_path = r"C:\Users\haith\PycharmProjects\automated disassembly\Data_and_codes\Images_with_Annotations\DATA\test\images\image_1.jpg"
model_path_screws = r"C:\Users\haith\PycharmProjects\automated disassembly\BB_to_Masks_Converter\best_screws.pt"
model_path_base_module = r"C:\Users\haith\PycharmProjects\automated disassembly\BB_to_Masks_Converter\best.pt"
# Save the output image
output_image_path = r"C:\Users\haith\PycharmProjects\automated disassembly\Data_and_codes\Images_with_Annotations\DATA\test\images\image_1_1.jpg"
output_image_modules=r"C:\Users\haith\PycharmProjects\automated disassembly\Data_and_codes\Images_with_Annotations\DATA\test\images\image_1_2.jpg"
# Defining Parts of the batteries:
#functional_parts= [Module, Base]
#fixing_parts= [screw]

# 1_ Generate RGB and depth data

RGB_np, depth_np= depth_generation.generate_rgb_and_depth(image_path)#------------first place to use the image path
cx, cy, fx, fy = depth_generation.get_camera_parameters(image_path) #------------second place to use the image path

# 2_Process screw detections
screws_masks_data, output_image = yolo_detectors.process_screw(RGB_np, model_path_screws)
cv2.imwrite(output_image_path, output_image)
module_base_masks_data, output_image_modules = yolo_detectors.process_modules(RGB_np, model_path_base_module)
cv2.imwrite(output_image_path, output_image_modules)
cv2.imshow("output_image", output_image)
cv2.imshow("output_image_modules", output_image_modules)
cv2.waitKey(0) 
cv2.destroyAllWindows()
# 3_Assign depth to masks and get camera parameters:
screws_depth_data = yolo_detectors.assign_depth_to_masks(screws_masks_data, depth_np)
module_base_depth_data= yolo_detectors.assign_depth_to_masks(module_base_masks_data, depth_np)

combined_data = depth_generation.combine_dictionaries_with_depth(
    screws_masks_data, screws_depth_data, 
    module_base_masks_data, module_base_depth_data
)

dictionary_of_screws = depth_generation.calculate_real_world_coordinates(screws_masks_data, screws_depth_data, cx, cy, fx, fy)
dictionary_of_objects_modules= depth_generation.calculate_real_world_coordinates(module_base_masks_data, module_base_depth_data, cx, cy, fx, fy)
combined_dictionary = dictionary_of_screws.copy()  # Make a copy to avoid modifying the original
combined_dictionary.update(dictionary_of_objects_modules)

#for key, values in dictionary_of_screws.items():
#    print(f"{key}: {values}")
result= objects_pipeline.detect_adjacency_with_sweep_planes(combined_dictionary)
#result = Topological_sorting.detect_adjacency(combined_data)
#Topological_sorting.visualize_results(combined_data, result)
Topological_sorting.save_results_to_sql_and_text(result, sql_file_path="results_1.db", text_file_path="results_1.txt")

for key, coordinates in combined_data.items():
    #Print the coordinates to understand their structure
    print(f"{key}")


depth_generation.print_sample_data (combined_data)


#cx, cy, fx, fy = depth_generation.get_camera_parameters(image_path) #------------second place to use the image path

#for key, depths in screws_masks_data.items():
#    print(f"{key}: {depths}")

# Print results
#for key, depths in screws_depth_data.items():
#    print(f"{key}: {depths}")

#print(f"Camera parameters:\n cx: {cx}, cy: {cy}, fx: {fx}, fy: {fy}")



#plt.imshow(RGB_np)
#plt.imshow(depth_np, cmap='jet', alpha=0.5)  # Overlay depth map on the RGB image
#plt.colorbar(label="Depth (m)")
#plt.axis('off')
#plt.show()






































# Define screws to check for intersection
#screw11 = "screw_11"
#screw5 = "screw_5"

# Ensure the screws exist in masks_data
#if screw11 in masks_data and screw5 in masks_data:
    # Get the pixel coordinates for both screws
#    pixels_screw11 = set(masks_data[screw11])
#    pixels_screw5 = set(masks_data[screw5])
    
    # Check for intersection
#    intersection = pixels_screw11 & pixels_screw5  # Set intersection
    
#    if intersection:
#        print(f"{screw11} and {screw5} intersect at {len(intersection)} pixel(s).")
#    else:
#        print(f"{screw11} and {screw5} do not intersect.")
#else:
#    print(f"One or both screws ({screw11}, {screw5}) are not found in masks_data.")

#print (masks_data)