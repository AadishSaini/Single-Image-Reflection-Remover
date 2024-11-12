# Simple Image Reflection Remover

A deep learning-based tool for removing reflections from images using a GCNet-Gan architecture. This project provides utilities for both generating synthetic reflection datasets and removing unwanted reflections from images. 

## ⭐ Features
- Synthetic reflection dataset generation
- Reflection removal using trained neural network
- HTML report generation with before/after comparisons
- PSNR and SSIM metrics calculation (when ground truth is available)
- Support for batch processing

## 🛠 Prerequisites
- Python 3.6+
- PyTorch
- OpenCV (cv2)
- NumPy
- PIL (Python Imaging Library)
- scikit-image
- tqdm

## 📥 Installation
1. Clone the repository:
bash
git clone https://github.com/yourusername/simple-image-reflection.git
cd simple-image-reflection


2. Install the required packages:
bash
pip install torch torchvision opencv-python numpy Pillow scikit-image tqdm


## 📁 Project Structure

simple-image-reflection/
├── base
├── bin
│   └── **pycache**
├── data_gen
├── ds
│   ├── input
├── images
│   └── defaultDataset
├── **pycache**
└── reflection
├── model.py
├── main.py
├── datagen.py
├── superimposer.py
└── trained_model_v4.pth


## 🎯 Usage
### 1. Generating Synthetic Dataset
To create a synthetic dataset with reflections, use the datagen.py script:
bash
python datagen.py --b path/to/base/images/ --r path/to/reflection/images/ --j path/to/output/


Arguments:
- --b: Directory containing base images
- --r: Directory containing reflection images
- --j: Output directory for the generated dataset

### 2. Removing Reflections
To remove reflections from images:
bash
python main.py 
 

Arguments:
- --dataset_name=: specify the name of the folder (ie the dataset) which is placed in the images folder. (if no parameter provided, the defaultDataset is used)

The script expects images to be organized in the following structure in the custom dataset.

images/
├── input/       # Input images with reflections


## 📤 Output
The program generates:
1. Processed images in the newly created output/ directory
2. An HTML report with side-by-side comparisons, in the dataset directory
3. PSNR and SSIM metrics (if ground truth is available)

## 🧠 Model Architecture
The reflection removal model uses a modified U-Net architecture with:
- Multi-scale feature extraction
- Skip connections
- Batch normalization
- LeakyReLU activation

## 📊 Results Visualization
After processing, an HTML report is automatically generated and opened in your default web browser, showing:
- Original images with reflections
- Processed images with reflections removed
- Quality metrics (if ground truth is available)

## ⚠ Known Limitations
- Input images are automatically padded to match model requirements
- Processing large images may require significant memory
- Best results are achieved with images similar to the training dataset

## 👥 Contributions
- Darshan Solanki, Team Captain
- Theodore R.
- Saksham Kumar
- Aadish Sasini
