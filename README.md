# Fruit Classification Prediction App

A production-ready Python application for fruit classification using Vision Transformer (ViT).

## Installation

```bash
pip install -r requirements.txt
```

## Model Installation

See [MODEL_INSTALLATION.md](MODEL_INSTALLATION.md) for instructions on how to install or train the model.

## Quick Start

### 1. Simple Prediction Script
```bash
python predict.py your_image.jpg
```

### 2. Command-Line Interface
```bash
python -m cli predict --image your_image.jpg
```

Batch prediction:
```bash
python -m cli predict-batch --directory ./images --output results.json
```

### 3. Python API
```python
from app import FruitClassifier

classifier = FruitClassifier(
    model_weights_path="model.pt",
    class_names_path="class_names.json"
)

result = classifier.predict("image.jpg", return_probs=True)
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")
```

### 4. Web Interface
```bash
streamlit run app.py
```

## Project Structure

```
app.py                    - Main prediction engine and web UI
cli.py                    - Command-line interface
predict.py                - Simple prediction script
class_names.json          - Fruit class names
requirements.txt          - Dependencies
setup.py                  - Package installation
__init__.py               - Package initialization
MODEL_INSTALLATION.md     - Model installation instructions
```

## Features

- Single and batch image predictions
- GPU support (automatic CUDA detection)
- Top-K predictions with confidence scores
- Four usage interfaces (CLI, script, Python API, Web UI)
- JSON output for batch results
- Production-ready error handling

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.0+
- Pillow, numpy, scikit-learn, pandas
- Streamlit for web interface
- Optional: CUDA 10.2+ for GPU support

## Model Details

- **Architecture**: Vision Transformer (google/vit-base-patch16-224)
- **Input**: 224×224 RGB images
- **Output**: Fruit class prediction with confidence score
- **Training Framework**: PyTorch + Hugging Face Transformers

## Usage Examples

### Single Image Prediction
```bash
python predict.py apple.jpg
```

### Batch Processing
```bash
python -m cli predict-batch --directory ./test_images --output predictions.json
```

### Extract Classes from Dataset
```bash
python -m cli save-classes --directory ./Training
```

## Configuration

Edit the model paths in code:
```python
classifier = FruitClassifier(
    model_weights_path="path/to/model.pt",
    class_names_path="path/to/classes.json",
    device='cuda'  # or 'cpu'
)
```

## Prediction Output

```python
{
    'class': 'Apple',
    'confidence': 0.95,
    'top_predictions': [
        {'class': 'Apple', 'probability': 0.95},
        {'class': 'Orange', 'probability': 0.03},
        ...
    ]
}
```

## Installation as Package

```bash
pip install -e .
```

Then use from anywhere:
```python
from fruit_classifier_app import FruitClassifier
```

## Performance Tips

- Use GPU for 10-100x faster predictions
- Process multiple images in batches for efficiency
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

## License

Built with Vision Transformer from Hugging Face Transformers.
