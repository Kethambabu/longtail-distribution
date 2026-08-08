# HAM10000 Skin Cancer Detection with GAN-based Data Augmentation

This project builds a deep learning pipeline for skin lesion classification using the HAM10000 dataset. Because the dataset is highly imbalanced, the workflow uses GAN-based synthetic image generation to create additional examples for minority classes before training a ResNet-18 classifier.

## Project goal

The main objective is to improve classification performance on rare skin disease categories by combining:

- synthetic image generation for class balancing
- a ResNet-18 classifier for multi-class prediction
- evaluation metrics such as accuracy, precision, recall, F1-score, confusion matrix, and FID

## What is included

- a Jupyter notebook for data preparation, GAN training, and classifier training
- a text version of the notebook code in notebook_code.txt
- project notes in interview_guide.md
- generated outputs such as synthetic samples, checkpoints, and plots

## Technologies used

- Python
- PyTorch and torchvision
- scikit-learn
- matplotlib and seaborn
- PIL, scipy, and tqdm
- kagglehub

## Project structure

- notebookd7f7a3492b (2).ipynb — main notebook
- notebook_code.txt — extracted script-style code from the notebook
- interview_guide.md — interview-ready summary of the project
- data/ — organized dataset folders
- train_original/ and test_original/ — train/test splits
- train_generated/ — synthetic images created by the GAN
- train_augmented/ — combined training data
- checkpoints/ — saved model checkpoints
- outputs/ — generated samples and plots

## Setup

1. Create and activate a Python environment.
2. Install the required packages:

   ```bash
   pip install -q kagglehub scikit-learn matplotlib seaborn pillow scipy tqdm
   ```

3. Open the notebook and run the cells in order.
4. Make sure you have access to a GPU if possible; training will be much faster than on CPU.

## Usage notes

- The notebook downloads the HAM10000 dataset through kagglehub.
- The workflow uses class-aware balancing and weighted sampling to improve performance on minority classes.
- Generated images, checkpoints, and training artifacts are stored in output folders that are ignored by Git.

## Expected outputs

- synthetic images for underrepresented classes
- FID-based quality evaluation for generated samples
- classifier performance metrics and confusion matrix
- saved checkpoints for later reuse

## Notes

This project is designed for experimentation and academic demonstration. For production use, you would typically add more validation, explainability tools, and stronger evaluation on external datasets.
