
# 3D Object Recognition & Spatial Relationship Analysis for Automated Disassembly

This repository contains a computer vision pipeline for **object recognition, depth estimation, and 3D spatial relationship analysis** aimed at **automated disassembly applications** (e.g. battery modules with screws, bases, and functional components).

The system combines **2D object detection**, **monocular depth estimation**, and **3D reasoning** to identify components and infer how they are physically related in space.

---

## 🚀 Project Overview

The pipeline performs the following steps:

1. **2D Object Detection**
   Detects screws, modules, and base components using YOLO models.

2. **Monocular Depth Estimation**
   Generates a dense depth map from a single RGB image using **Depth Pro**.

3. **3D Reconstruction**
   Converts pixel-level detections into **real-world 3D coordinates** using camera intrinsics.

4. **Spatial Relationship Reasoning**
   Determines adjacency, coverage, and fixation relationships between components using sweep-plane–based algorithms.

5. **Structured Output**
   Exports results as:

   * SQL database
   * Human-readable text file

---

## 📁 Repository Structure

```
├── Main.py
├── depth_generation.py
├── objects_pipeline.py
├── Topological_sorting.py
├── yolo_detectors.py
├── README.md
```

### Key Files

* **Main.py**
  Entry point. Runs the full pipeline from image input to relationship extraction and result storage.

* **depth_generation.py**

  * RGB + depth generation (Depth Pro)
  * Camera parameter estimation
  * Depth assignment to segmentation masks
  * Conversion from image coordinates to real-world 3D coordinates

* **objects_pipeline.py**

  * Improved sweep-plane algorithm
  * Robust detection of spatial relationships (+X, −X, +Y, −Y)
  * Fixation detection (e.g. screws fixing functional parts)

* **Topological_sorting.py**

  * Alternative adjacency detection
  * 3D visualization of components and relationships
  * Saving results to SQL and text formats

---

## 🧠 Detected Relationships

For each component, the system infers:

* `isFixedBy_minusZ` → Fixing parts (e.g. screws)
* `isDirectCoveredBy_plusX`
* `isDirectCoveredBy_minusX`
* `isDirectCoveredBy_plusY`
* `isDirectCoveredBy_minusY`

These relationships are crucial for **automated disassembly planning**.

---

## ⚙️ Requirements

* Python 3.9+
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Pandas
* SQLite
* Ultralytics YOLO
* Depth Pro (pretrained checkpoint required)

> ⚠️ Model checkpoints and dataset images are **not included** in this repository.

---

## ▶️ How to Run

1. Update paths in `Main.py`:

   * Input image
   * YOLO model checkpoints
   * Output directories

2. Run the pipeline:

   ```bash
   python Main.py
   ```

3. Outputs:

   * Annotated images
   * `results.db` (SQL database)
   * `results.txt` (readable summary)

---

## 📊 Output Example

Each object is stored with:

* Class and ID
* Spatial relationships
* Fixation constraints

Example:

```
Object: module_1
Class: module
Is Fixed By (-Z): screw_3, screw_5
Directly Covered By (+X): module_2
```

---

## 🔬 Application Context

This project is designed for:

* Automated disassembly
* Robotic manipulation planning
* 3D scene understanding from monocular images
* Industrial inspection and recycling systems

---

## 📌 Notes

* Camera intrinsics are approximated when not available.
* Depth values are scaled and normalized for real-world interpretation.
* Algorithms are robust to segmentation noise using tolerance thresholds.

---

## 📜 License

This project is intended for **research and academic use**.
Please contact me for commercial licensing.


Just say the word 🙂
