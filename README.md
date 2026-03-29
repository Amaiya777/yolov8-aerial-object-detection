# Aerial Object Detection using YOLOv8

##  Overview
This project implements an object detection system for aerial imagery using YOLOv8. The model is trained on the FAIR1M dataset to detect multiple real-world objects such as aircraft, ships, vehicles, and infrastructure elements.

---

##  Features
- Multi-class object detection (22 classes)
- Custom dataset configuration using YAML
- GPU-accelerated training (NVIDIA DGX A100)
- Efficient detection using YOLOv8 architecture
- Organized training pipeline with validation support

---

##  Model
- Model: YOLOv8 (Ultralytics)
- Framework: PyTorch
- Task: Object Detection

---

##  Dataset
- Dataset: FAIR1M (Aerial Image Dataset)
- Total Classes: 22
- Example classes:
  - Aircraft
  - Ship
  - Vehicle
  - Bridge
  - Windmill
  - Solar Panel

>  Dataset not included in this repository due to large size.

---

##  Tech Stack
- Python
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy, Matplotlib
- YAML (Custom dataset configuration)

---

##  Training
- Training performed on NVIDIA DGX A100 GPU Pod
- Custom YAML used for dataset configuration
- Training includes validation monitoring

---

##  Results
- Successfully trained model for multi-class detection
- Model capable of detecting objects in aerial imagery with good accuracy
- Outputs include bounding boxes and class predictions
