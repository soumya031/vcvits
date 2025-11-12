# Fruit Classification App - Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Web Interface
To launch the interactive web application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### 2. Simple Prediction Script
```bash
python predict.py path/to/image.jpg
```

### 3. Command-Line Interface
```bash
# Single image prediction
python -m cli predict --image path/to/image.jpg

# Batch processing
python -m cli predict-batch --directory path/to/images --output results.json

# Extract class names from training directory
python -m cli save-classes --directory path/to/Training
```

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
├── cli.py                 # Command-line interface
├── predict.py             # Simple prediction script
├── class_names.json       # List of fruit classes
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
├── __init__.py           # Package initialization
├── MODEL_INSTALLATION.md # Model installation instructions
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
