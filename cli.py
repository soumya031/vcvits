#!/usr/bin/env python3
"""
Command-line interface for fruit classification.
Usage: 
    python -m cli predict --image image.jpg
    python -m cli predict-batch --directory ./images --output results.json
    python -m cli save-classes --directory ./Training
"""

import argparse
import os
import json
import sys
from app import FruitClassifier, save_class_names

def predict_command(args):
    """Handle single image prediction."""
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return 1
    
    try:
        print("Loading fruit classifier model...")
        classifier = FruitClassifier(
            model_weights_path=args.model,
            class_names_path=args.classes
        )
        
        print(f"Analyzing image: {args.image}")
        result = classifier.predict(args.image, return_probs=not args.no_probs)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Predicted Fruit: {result['class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            if 'top_predictions' in result:
                print("\nTop Predictions:")
                for i, pred in enumerate(result['top_predictions'], 1):
                    print(f"  {i}. {pred['class']:<20} {pred['probability']:.2%}")
        
        return 0
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return 1

def predict_batch_command(args):
    """Handle batch image prediction."""
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' not found.")
        return 1
    
    try:
        print("Loading fruit classifier model...")
        classifier = FruitClassifier(
            model_weights_path=args.model,
            class_names_path=args.classes
        )
        
        print(f"Processing images from: {args.directory}")
        results = classifier.predict_batch(args.directory, args.extension)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        return 0
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        return 1

def save_classes_command(args):
    """Save class names from training directory."""
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' not found.")
        return 1
    
    try:
        # Get class names from directory structure
        class_names = sorted([d for d in os.listdir(args.directory) 
                             if os.path.isdir(os.path.join(args.directory, d))])
        
        if not class_names:
            print("Warning: No subdirectories found in training directory.")
            return 1
        
        save_class_names(class_names, args.output)
        print(f"Found {len(class_names)} classes.")
        return 0
    except Exception as e:
        print(f"Error saving class names: {str(e)}")
        return 1

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fruit Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='Fruit Classifier 1.0'
    )
    
    subparsers = parser.add_subparsers(
        title='Commands',
        dest='command',
        help='Available commands'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict fruit class for a single image'
    )
    predict_parser.add_argument(
        '--image',
        required=True,
        help='Path to image file'
    )
    predict_parser.add_argument(
        '--model',
        default='model.pt',
        help='Path to model weights file'
    )
    predict_parser.add_argument(
        '--classes',
        default='class_names.json',
        help='Path to class names file'
    )
    predict_parser.add_argument(
        '--no-probs',
        action='store_true',
        help='Do not return probability distribution'
    )
    predict_parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    predict_parser.set_defaults(func=predict_command)
    
    # Batch predict command
    batch_parser = subparsers.add_parser(
        'predict-batch',
        help='Predict fruit classes for multiple images'
    )
    batch_parser.add_argument(
        '--directory',
        required=True,
        help='Directory containing images'
    )
    batch_parser.add_argument(
        '--extension',
        default='.jpg',
        help='Image file extension (default: .jpg)'
    )
    batch_parser.add_argument(
        '--output',
        help='Output JSON file path'
    )
    batch_parser.add_argument(
        '--model',
        default='model.pt',
        help='Path to model weights file'
    )
    batch_parser.add_argument(
        '--classes',
        default='class_names.json',
        help='Path to class names file'
    )
    batch_parser.set_defaults(func=predict_batch_command)
    
    # Save classes command
    save_parser = subparsers.add_parser(
        'save-classes',
        help='Save class names from training directory structure'
    )
    save_parser.add_argument(
        '--directory',
        required=True,
        help='Training directory with class subdirectories'
    )
    save_parser.add_argument(
        '--output',
        default='class_names.json',
        help='Output JSON file path'
    )
    save_parser.set_defaults(func=save_classes_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())