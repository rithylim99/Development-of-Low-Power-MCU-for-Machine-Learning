# Development of Low Power MCU for Machine Learning
---
# I. Tensorflow Setup for PC
## Setup Tensorflow that support with GPU on window
For more detail, you cacn follow (https://www.tensorflow.org/install/pip)

## 1. System Requirement
- TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows. Starting with TensorFlow 2.11, you will need to install TensorFlow in WSL2, or install tensorflow or tensorflow-cpu and, optionally, try the TensorFlow-DirectML-Plugin
- Windows 7 or higher (64-bit)
## 2. Install Microsoft Visual C++ Redistributable
Install the Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, and 2019. Starting with the TensorFlow 2.1.0 version, the msvcp140_1.dll file is required from this package (which may not be provided from older redistributable packages). The redistributable comes with Visual Studio 2019 but can be installed separately:
- Go to the Microsoft Visual C++ downloads.
- Scroll down the page to the Visual Studio 2015, 2017 and 2019 section.
- Download and install the Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 for your platform.

Make sure long paths are enabled on Windows.
## 3. Install Miniconda
Miniconda is the recommended approach for installing TensorFlow with GPU support. It creates a separate environment to avoid changing any installed software in your system. This is also the easiest way to install the required software especially for the GPU setup.

Download the Miniconda Windows Installer by the link below: (https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).

## 4. Create a conda environment
Create a new conda environment by open `Anaconda Prompt` and named `python39` with the following command.
```
conda create --name python39 python=3.9
```
Deactivate and activate it with the following commands.
```
conda deactivate
conda activate python39
```
Make sure it is activated for the rest of the installation.

## 5. GPU Setup
If you don't have GPU driver you can install the driver first, otherwise we can start install the CUDA, cuDNN with conda.
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
## 6. Install the Tensorflow (Version below 2.11)
```
pip install "tensorflow<2.11" 
```
We can verify the setup by typing `python` using cammand:
```
>> import tensorflow as tf
>> tf.config.list_physical_devices('GPU')

>> tf.test.is_gpu_available()
```
If there are any error with `Numpy` package version, you can uninstall the `Numpy` and reinstall the `Numpy` by version 1.26.4
```
pip uninstall numpy
pip install "numpy==1.26.4"
```