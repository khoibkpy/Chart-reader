# Automated Chart Data Extraction System

A hybrid deep learning and computer vision pipeline designed to automatically classify 2D bar charts and pie charts, extract their key elements, and accurately reconstruct the underlying visual data into structured JSON formats.

## Key Features & Performance
* **High Accuracy:** Achieves a **98% classification accuracy** on standard 2D bar and pie chart images.
* **Hybrid Pipeline:** Combines State-of-the-Art object detection with robust optical character recognition (OCR).
* **Unsupervised Mapping:** Utilizes spatial clustering to dynamically pair numerical data points with their respective text labels.
* **Structured Output:** Automatically exports raw visual components into organized JSON structures for easy database migration.

---

## System Architecture

The pipeline processes input chart images through three distinct phases:

1. **Element Detection (YOLOv8):** Scans the image to localize and classify structural chart elements (bars, pie slices, legends, and axes).
2. **Text Transcription (PaddleOCR):** Extracts textual data from the image, isolating titles, values, and axis labels.
3. **Data Reconstruction (K-Means Clustering):** Applies spatial clustering algorithms to mathematically calculate the proximity between detected chart shapes and text blocks, mapping the data accurately before serializing it into JSON.

---

## Tech Stack
* **Language:** Python
* **Deep Learning & CV:** PyTorch, Ultralytics (YOLOv8), PaddleOCR, OpenCV
* **Data & Clustering:** NumPy, Scikit-learn (K-Means)
