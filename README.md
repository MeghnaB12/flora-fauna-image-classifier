# 🌿 Flora & Fauna Image Classifier (ViT-Huge)

This project contains the complete code for a deep learning model designed to classify images of flora and fauna into 10 distinct categories. The model is built using PyTorch and leverages state-of-the-art Transfer Learning.

## 🚀 Key Features & Goals

The pipeline is engineered for high performance in a competitive setting, utilizing **Mixed Precision Training** and **Test Time Augmentation** principles to maximize accuracy.

* **SOTA Architecture:** Utilizes the massive `vit_h_14` (Vision Transformer Huge) pre-trained on ImageNet.
* **Robustness:** Achieves generalization through heavy data augmentation (Rotation, Flips, Crops).
* **Efficiency:** Implements Gradient Scaling (AMP) to handle the massive model footprint on consumer GPUs.

## 📈 Methodology

The core of this solution is **Transfer Learning** with a Vision Transformer. Instead of training a massive network from scratch, we leverage a pre-trained backbone to extract high-level semantic features from the images.



### 1. Image Preprocessing (Augmentation)
To prevent overfitting and ensure the model recognizes objects in various orientations, the training data undergoes a rigorous transformation pipeline:
* **Resizing:** Images are resized to 224x224 using Bicubic interpolation.
* **Geometric Transforms:**
    * Random Horizontal Flips (50% probability)
    * Random Vertical Flips (50% probability)
    * Random Rotation (±30 degrees)
* **Cropping:** Random cropping with padding is applied to simulate shifting viewpoints.
* **Normalization:** Pixel values are normalized using standard ImageNet mean and standard deviation.

### 2. The Model (Vision Transformer)
The model used is `vit_h_14` (Vision Transformer Huge) loaded via `torchvision.models`.
* **Backbone:** The weights are initialized from `IMAGENET1K_SWAG_LINEAR_V1`.
* **Freezing:** The heavy feature extraction layers are frozen (`requires_grad = False`) to preserve learned filters.
* **Custom Head:** The original classification head is replaced with a custom fine-tuning block:
    * Linear Layer (1280 $\rightarrow$ 128)
    * Batch Normalization
    * GELU Activation
    * Dropout (0.25)
    * Final Output (128 $\rightarrow$ 10 Classes)

### 3. Training & Inference
* **Optimizer:** NAdam optimizer (Learning Rate: 0.001).
* **Loss Function:** CrossEntropyLoss.
* **Mixed Precision:** Uses `torch.cuda.amp.GradScaler` to train in FP16, speeding up training and reducing VRAM usage.
* **Inference:** The notebook includes a `classify_images_to_csv` function that processes the test folder and generates a `submission.csv`.

## 🛠️ Tech Stack

* **Core:** Python 3
* **Deep Learning:** PyTorch, Torchvision
* **Model Architecture:** Vision Transformer (ViT-Huge)
* **Data Handling:** Pandas, NumPy, PIL
* **Metrics:** Scikit-learn (F1 Score)
* **Utilities:** tqdm (Progress bars)

## 🏃 Running the Project

### 1. Dependencies
It is recommended to run this in a Kaggle environment or a local virtual environment with GPU support.

```bash
pip install torch torchvision pandas numpy scikit-learn tqdm pillow
```

### 2. Dataset

This model was trained on the **Flora & Fauna** dataset as part of a university challenge. Due to privacy and access restrictions, the dataset is not publicly available and is not included in this repository.

Therefore, the script cannot be run out-of-the-box without downloading the specific competition data separately and placing it in the correct directory structure.

### 3. Notebook Review

The image_classifier.ipynb notebook contains the full, end-to-end code for the methodology. It serves as a reference implementation for ViT Transfer Learning, including:

* Augmentation: Complex torchvision.transforms pipelines.

* Model Definition: How to modify ViT heads and freeze backbones.

* Training Loop: A complete training loop implementation with GradScaler for Mixed Precision.

* Inference: Logic to iterate over a test folder and generate a submission CSV.

This notebook can be reviewed to understand the complete logic and architecture, but cannot be executed without the original dataset.
