# Fridge Detection Model

This project implements a YOLOv8-based object detection model for identifying food items in fridge images. The model is trained to recognize 30 different food categories commonly found in refrigerators.

## Features

- **30 Food Categories**: Recognizes apple, banana, beef, blueberries, bread, butter, carrot, cheese, chicken, chicken_breast, chocolate, corn, eggs, flour, goat_cheese, green_beans, ground_beef, ham, heavy_cream, lime, milk, mushrooms, onion, potato, shrimp, spinach, strawberries, sugar, sweet_potato, and tomato
- **Data Augmentation**: Implements comprehensive data augmentation techniques to improve model robustness
- **Model Comparison**: Trains both pretrained and from-scratch YOLOv8 models for performance comparison
- **Visualization**: Provides detailed visualizations of class distributions, augmentations, and results

## Dataset

The model uses the "aicook" dataset from Roboflow, which contains labeled images of food items in refrigerator settings. The dataset includes:
- Training images with bounding box annotations
- Class distribution analysis and balancing
- Augmented samples for improved training

## Requirements

See `requirements.txt` for a complete list of dependencies. Main packages include:
- `ultralytics` - YOLOv8 implementation
- `roboflow` - Dataset management
- `albumentations` - Data augmentation
- `opencv-python` - Image processing
- `matplotlib` - Visualization
- `torch` - Deep learning framework

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/wyattmog/Fridge-Detection-Model.git
   cd Fridge-Detection-Model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Roboflow API key:
   ```bash
   export ROBOFLOW_API_KEY="your_api_key_here"
   ```

4. Run the Jupyter notebook:
   ```bash
   jupyter notebook Fridge_Detection_Model.ipynb
   ```

## Usage

The notebook is organized into the following sections:

1. **Data Loading**: Downloads the dataset from Roboflow
2. **Data Analysis**: Visualizes class distributions and sample images
3. **Data Augmentation**: Implements cropping and albumentations-based augmentation
4. **Model Training**: Trains both pretrained and from-scratch YOLOv8 models
5. **Evaluation**: Compares model performance and generates predictions
6. **Visualization**: Creates plots and diagrams showing the ML pipeline

## Model Performance

The notebook trains two models:
- **Pretrained YOLOv8n**: Fine-tuned from COCO pretrained weights
- **From-scratch YOLOv8n**: Trained without pretrained weights

Results include precision, recall, and mAP metrics for both approaches, allowing for performance comparison.

## Data Augmentation Techniques

The project implements several augmentation strategies:

1. **Class Balancing**: Crops objects from existing images to balance class distribution
2. **Albumentations Pipeline**:
   - Horizontal flipping
   - Random rotation (±10°)
   - Brightness/contrast adjustment
   - Gaussian blur and noise
   - Coarse dropout for occlusion simulation

## Output

The trained models and results are saved in the `runs/train/` directory:
- `fridge_pretrained/`: Pretrained model results
- `fridge_scratch/`: From-scratch model results

Each directory contains:
- Model weights (`best.pt`, `last.pt`)
- Training metrics and plots
- Validation results

## Environment Information

The notebook includes system information logging to track:
- Python and package versions
- GPU availability and specifications
- CUDA version (if available)

## License

This project is open source and available under the [MIT License](LICENSE).