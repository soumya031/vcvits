# Fruit Classification App - Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

To launch the interactive web application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Features

### 1. Single Image Prediction
- Upload a fruit image (JPG, PNG, BMP)
- Get instant classification with confidence score
- View top 5 predictions with probabilities

### 2. Batch Processing
- Process multiple fruit images from a directory
- Get summary statistics (total images, average confidence, unique fruits)
- Download results as JSON file

### 3. About Page
- Information about the model
- Technology stack details
- Tips for best results

## File Structure

```
.
├── app.py                 # Main application (all-in-one)
├── class_names.json       # List of fruit classes
├── requirements.txt       # Python dependencies
└── model.pt              # Trained model weights (optional)
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Streamlit
- PIL/Pillow
- numpy, scikit-learn, pandas

## Notes

- GPU acceleration is automatically detected and used if available
- If `model.pt` is not found, the app will use a pre-trained Vision Transformer
- Images are automatically resized to 224×224 pixels
- Works with JPG, PNG, and BMP formats

## Troubleshooting

**Issue**: Model loading is slow
- This is normal on first run as it downloads the Vision Transformer model
- Subsequent runs will be faster

**Issue**: "Image not found" error in batch processing
- Make sure the directory path exists
- Ensure image files have the correct extension (jpg, jpeg, png, bmp)

**Issue**: Out of memory error
- Try using CPU mode or process smaller batches
- Reduce image batch size if needed
