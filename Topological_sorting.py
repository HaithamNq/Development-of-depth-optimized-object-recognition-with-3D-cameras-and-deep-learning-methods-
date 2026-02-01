import cv2 
import os
import sys 
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yolo_detectors
import depth_generation
import sqlite3
import pandas as pd


def extract_min_max_from_mask_points(dictionary_of_coordinates):
    """
    Extract min and max values for x, y, and z coordinates from the input dictionary.

    Args:
        dictionary_of_coordinates: Dictionary where keys are object names and values are lists of tuples (X, Y, Z)
                                   representing the real-world camera coordinates of each point in the mask.

    Returns:
        Dictionary with min and max values for each object.
    """
    bounds = {}
    for name, points in dictionary_of_coordinates.items():
        points_array = np.array(points, dtype=float)
        x_min, x_max = points_array[:, 0].min(), points_array[:, 0].max()
        y_min, y_max = points_array[:, 1].min(), points_array[:, 1].max()
        z_min, z_max = points_array[:, 2].min(), points_array[:, 2].max()

        bounds[name] = {
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(y_min), "y_max": float(y_max),
            "z_min": float(z_min), "z_max": float(z_max)
        }

    return bounds



def detect_adjacency(dictionary_of_coordinates):
    """
    Use sweep-line algorithm to detect adjacency and overlaps in x, y, z directions.
    Handles fixing and functional parts separately for `isFixedBy_minusZ`.

    Args:
        dictionary_of_coordinates: Dictionary where keys are object names and values are lists of tuples (X, Y, Z)
                                   representing the real-world camera coordinates of each point in the mask.

    Returns:
        component_dictionary_layers: A dictionary containing adjacency relationships.
    """
    # Step 1: Extract min and max bounds for each object
    bounds = extract_min_max_from_mask_points(dictionary_of_coordinates)

    fixing_parts = ["screw"]
    functional_parts = ["module", "base", "object"]

    components = {}
    objects = [
        {
            "name": name,
            **bounds[name]
        }
        for name in bounds
    ]


    for obj1 in objects:
        obj1_class = obj1["name"].split("_")[0].lower()

        # Functional parts processing
        if obj1_class in functional_parts:
            component = {
                "new_component": "True",
                "class": obj1_class,
                "id": int(obj1["name"].split("_")[-1]),
                "name": obj1["name"],
                "isFixedBy_minusZ": [],  # Initialize as empty, will be populated if fixing parts overlap
                "isDirectCoveredBy_plusX": [],
                "isDirectCoveredBy_plusY": [],
                "isDirectCoveredBy_minusX": [],
                "isDirectCoveredBy_minusY": []
            }

            for obj2 in objects:
                
                if obj1 == obj2:
                    continue

                obj2_class = obj2["name"].split("_")[0].lower()
                if obj2_class in functional_parts:
                # Check z-axis overlap
                    z_overlap = max(obj1['z_min'], obj2['z_min']) <= min(obj1['z_max'], obj2['z_max'])

                    if z_overlap:
                        # Check y-axis overlap and direction
                        y_overlap = max(obj1['y_min'], obj2['y_min']) <= min(obj1['y_max'], obj2['y_max'])
                        if y_overlap:
                            if obj1['x_max'] <= obj2['x_min']:
                                component["isDirectCoveredBy_plusX"].append(obj2['name'])
                            if obj2['x_max'] <= obj1['x_min']:
                                component["isDirectCoveredBy_minusX"].append(obj2['name'])

                    # Check x-axis overlap and direction
                        x_overlap = max(obj1['x_min'], obj2['x_min']) <= min(obj1['x_max'], obj2['x_max'])
                        if x_overlap:
                            if obj1['y_max'] <= obj2['y_min']:
                             component["isDirectCoveredBy_plusY"].append(obj2['name'])
                            if obj2['y_max'] <= obj1['y_min']:
                                component["isDirectCoveredBy_minusY"].append(obj2['name'])

                components[obj1["name"]] = component

        # Fixing parts processing
        elif obj1_class in fixing_parts:
            component = {
                "new_component": "True",
                "class": obj1_class,
                "id": int(obj1["name"].split("_")[-1]) + 1000,  # ID offset for fixing parts
                "name": obj1["name"],
                "possibleDisassembly_minusZ": ["True"]
            }

            for obj2 in objects:
                obj2_class = obj2["name"].split("_")[0].lower()

                # Check if obj2 is in functional parts and functional bounds enclose fixing bounds
                if obj2_class in functional_parts:
                    if (
                        obj2['x_min'] <= obj1['x_min'] and
                        obj1['x_max'] <= obj2['x_max'] and
                        obj2['y_min'] <= obj1['y_min'] and
                        obj1['y_max'] <= obj2['y_max']
                    ):
                        if "isFixedBy_minusZ" not in components.get(obj2["name"], {}):
                            if obj2["name"] not in components:
                                components[obj2["name"]] = {
                                    "new_component": "True",
                                    "class": obj2_class,
                                    "id": int(obj2["name"].split("_")[-1]),
                                    "name": obj2["name"],
                                    "isFixedBy_minusZ": [],
                                    "isDirectCoveredBy_plusX": [],
                                    "isDirectCoveredBy_plusY": [],
                                    "isDirectCoveredBy_minusX": [],
                                    "isDirectCoveredBy_minusY": []
                                }
                            components[obj2["name"]]["isFixedBy_minusZ"] = []
                        components[obj2["name"]]["isFixedBy_minusZ"].append(obj1["name"])

            components[obj1["name"]] = component

    return components

# "isDirectCoveredBy_plusX": [id number instead of name], immer auf die {} achten, 
#--------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def visualize_results(dictionary_of_coordinates, results):
    """
    Visualize the bounding boxes and relationships between objects in 3D space.

    Args:
        dictionary_of_coordinates: Dictionary where keys are object names and values are lists of tuples (X, Y, Z).
        results: The output dictionary containing adjacency relationships.
    """
    # Generate a figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {"module": "blue", "base": "green", "screw": "red"}
    edges = {"isDirectCoveredBy_plusX": "red", "isDirectCoveredBy_minusX": "blue",
             "isDirectCoveredBy_plusY": "orange", "isDirectCoveredBy_minusY": "purple"}
    
    # Plot each object's bounding box
    for obj_name, coords in dictionary_of_coordinates.items():
        x, y, z = zip(*coords)
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        z_min, z_max = min(z), max(z)
        
        obj_class = obj_name.split("_")[0].lower()
        color = colors.get(obj_class, "black")
        
        # Draw the bounding box
        vertices = [
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        faces = [
            [vertices[i] for i in [0, 1, 2, 3]],  # Bottom face
            [vertices[i] for i in [4, 5, 6, 7]],  # Top face
            [vertices[i] for i in [0, 1, 5, 4]],  # Front face
            [vertices[i] for i in [2, 3, 7, 6]],  # Back face
            [vertices[i] for i in [1, 2, 6, 5]],  # Right face
            [vertices[i] for i in [0, 3, 7, 4]]   # Left face
        ]
        poly3d = Poly3DCollection(faces, alpha=0.3, facecolors=color, edgecolor="k")
        ax.add_collection3d(poly3d)
        
        # Annotate the object
        ax.text(np.mean([x_min, x_max]), np.mean([y_min, y_max]), np.mean([z_min, z_max]),
                obj_name, color=color, fontsize=10)
    
    # Draw relationships
    for obj_name, relationships in results.items():
        if "isDirectCoveredBy_plusX" in relationships:
            for target in relationships["isDirectCoveredBy_plusX"]:
                draw_arrow(ax, dictionary_of_coordinates[obj_name], dictionary_of_coordinates[target], "x", edges["isDirectCoveredBy_plusX"])
        if "isDirectCoveredBy_minusX" in relationships:
            for target in relationships["isDirectCoveredBy_minusX"]:
                draw_arrow(ax, dictionary_of_coordinates[obj_name], dictionary_of_coordinates[target], "x", edges["isDirectCoveredBy_minusX"])
        if "isDirectCoveredBy_plusY" in relationships:
            for target in relationships["isDirectCoveredBy_plusY"]:
                draw_arrow(ax, dictionary_of_coordinates[obj_name], dictionary_of_coordinates[target], "y", edges["isDirectCoveredBy_plusY"])
        if "isDirectCoveredBy_minusY" in relationships:
            for target in relationships["isDirectCoveredBy_minusY"]:
                draw_arrow(ax, dictionary_of_coordinates[obj_name], dictionary_of_coordinates[target], "y", edges["isDirectCoveredBy_minusY"])
        if "isFixedBy_minusZ" in relationships:
            for target in relationships["isFixedBy_minusZ"]:
                draw_arrow(ax, dictionary_of_coordinates[obj_name], dictionary_of_coordinates[target], "z", "green")
    
    # Set plot labels and limits
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.title("3D Visualization of Object Relationships")
    plt.show()

def draw_arrow(ax, source_coords, target_coords, axis, color):
    """
    Draw an arrow between two objects along a specified axis.

    Args:
        ax: The matplotlib 3D axis.
        source_coords: List of (X, Y, Z) coordinates of the source object.
        target_coords: List of (X, Y, Z) coordinates of the target object.
        axis: The axis of the relationship ('x', 'y', 'z').
        color: The color of the arrow.
    """
    source_x, source_y, source_z = zip(*source_coords)
    target_x, target_y, target_z = zip(*target_coords)
    
    # Arrow starts and ends
    start = [np.mean(source_x), np.mean(source_y), np.mean(source_z)]
    end = [np.mean(target_x), np.mean(target_y), np.mean(target_z)]
    
    if axis == "x":
        start[1:] = [end[1], end[2]]  # Align Y and Z
    elif axis == "y":
        start[::2] = [end[0], end[2]]  # Align X and Z
    elif axis == "z":
        start[:2] = [end[0], end[1]]  # Align X and Y
    
    ax.quiver(*start, *(np.array(end) - np.array(start)), color=color, arrow_length_ratio=0.1)

def save_results_to_sql_and_text(results, sql_file_path="results.db", text_file_path="results.txt"):
    """
    Save the adjacency results to an SQL database and a text file.

    Args:
        results: The dictionary containing adjacency results.
        sql_file_path: Path to save the SQL database.
        text_file_path: Path to save the text file.
    """
    import sqlite3

    # Default keys for all objects
    default_keys = {
        "isFixedBy_minusZ": [],
        "isDirectCoveredBy_plusX": [],
        "isDirectCoveredBy_plusY": [],
        "isDirectCoveredBy_minusX": [],
        "isDirectCoveredBy_minusY": []
    }

    # Prepare records with all keys
    records = []
    for obj_name, obj_data in results.items():
        # Ensure all default keys are present
        for key, default_value in default_keys.items():
            obj_data.setdefault(key, default_value)

        # Add the object data to the records
        record = {
            "name": obj_name,
            "class": obj_data["class"],
            "id": obj_data["id"],
            "new_component": obj_data["new_component"],
            "isFixedBy_minusZ": ", ".join(obj_data["isFixedBy_minusZ"]),
            "isDirectCoveredBy_plusX": ", ".join(obj_data["isDirectCoveredBy_plusX"]),
            "isDirectCoveredBy_plusY": ", ".join(obj_data["isDirectCoveredBy_plusY"]),
            "isDirectCoveredBy_minusX": ", ".join(obj_data["isDirectCoveredBy_minusX"]),
            "isDirectCoveredBy_minusY": ", ".join(obj_data["isDirectCoveredBy_minusY"]),
        }
        records.append(record)

    # Create a DataFrame
    import pandas as pd
    df = pd.DataFrame(records)

    # Save to SQL
    conn = sqlite3.connect(sql_file_path)
    df.to_sql("adjacency_results", conn, if_exists="replace", index=False)
    conn.close()

    # Save to Text File
    with open(text_file_path, "w") as text_file:
        for record in records:
            text_file.write(f"Object: {record['name']}\n")
            text_file.write(f"  Class: {record['class']}\n")
            text_file.write(f"  ID: {record['id']}\n")
            text_file.write(f"  New Component: {record['new_component']}\n")
            text_file.write(f"  Is Fixed By (-Z): {record['isFixedBy_minusZ']}\n")
            text_file.write(f"  Directly Covered By (+X): {record['isDirectCoveredBy_plusX']}\n")
            text_file.write(f"  Directly Covered By (+Y): {record['isDirectCoveredBy_plusY']}\n")
            text_file.write(f"  Directly Covered By (-X): {record['isDirectCoveredBy_minusX']}\n")
            text_file.write(f"  Directly Covered By (-Y): {record['isDirectCoveredBy_minusY']}\n")
            text_file.write("\n")
    print(f"Results saved to SQL at {sql_file_path} and text file at {text_file_path}")




# Example Usage
#if __name__ == "__main__":
    # Example dictionary of coordinates
#    dictionary_of_coordinates = {
#    "Module_1": [(1, 2, 3), (2, 2, 4), (3, 2, 5)],
#    "module_2": [(2, 5, 3), (3, 2, 4), (4, 2, 5)],
#    "module_5": [(10, 5, 3), (8, 2, 4), (6, 2, 5)],
#    "screw_1": [(1.5, 2.5, 3), (2, 3, 7)],
#    "screw_2": [(3.5, 4.5, 5), (4, 5, 5.5)]
#}


