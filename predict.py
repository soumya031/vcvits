#!/usr/bin/env python3
"""
Simple prediction script for fruit classification.
Usage: python predict.py image.jpg
"""

import sys
import os
import json
from PIL import Image
import torch
from app import FruitClassifier

def main():
    """Main prediction function."""
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py apple.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    try:
        # Initialize classifier
        print("Loading fruit classifier model...")
        classifier = FruitClassifier(
            model_weights_path="model.pt",
            class_names_path="class_names.json"
        )
        
        # Make prediction
        print(f"Analyzing image: {image_path}")
        result = classifier.predict(image_path, return_probs=True)
        
        # Display results
        print("\n" + "="*50)
        print("FRUIT CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Predicted Fruit: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if 'top_predictions' in result:
            print("\nTop 5 Predictions:")
            print("-" * 30)
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"{i}. {pred['class']:<20} {pred['probability']:.2%}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()