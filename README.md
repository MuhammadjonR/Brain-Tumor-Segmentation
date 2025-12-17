# Brain Tumor Segmentation from MRI

This project presents a comparative analysis of traditional image segmentation methods and a deep learning-based U-Net model for brain tumor segmentation from MRI images.

## Methods Implemented
- Thresholding (Otsu, Adaptive)
- K-means Clustering (k=3,4)
- Region Growing
- Watershed Segmentation
- Morphological Skull Stripping
- Edge Detection (Canny, Sobel, LoG)
- Active Contour (Snakes)
- Chan-Vese Segmentation
- U-Net Deep Learning Model

## Application
An interactive Streamlit web application is developed to visualize and compare segmentation results in real time.

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
