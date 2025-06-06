# Unsupervised Domain Adaptation Learning

This repository contains implementations of unsupervised domain adaptation techniques using Gradient Reversal Layer (GRL) and PaSST feature extractors across various datasets. The code was collected and modified from various GitHub sources as a learning exercise and precursor to our main research project.

## Project Overview

This work serves as foundational research for our summer internship project at **IIT Mandi**: *Domain Adaptation for Acoustic Scene Classification for different devices using Gradual Vanishing Bridge*.

## Repository Structure

```
├── notebooks/
│   ├── grl_dcase.ipynb              # GRL implementation on DCASE dataset
│   ├── grl_mnist-svhm.ipynb         # GRL implementation on MNIST-SVHN datasets
│   └── passt_pretrained_practise.ipynb # PaSST feature extractor practice
├── datasets/
│   ├── dcase/                       # DCASE TAU audio files (to be downloaded)
│   │   └── audio/                   # Contains acoustic scene audio files
│   └── audio_files/                 # Sample audio files for PaSST practice
└── .gitignore
```

## Implemented Techniques

### 1. Gradient Reversal Layer (GRL/DANN)
- Implementation on MNIST ↔ SVHN domain adaptation in [`grl_mnist-svhm.ipynb`](notebooks/grl_mnist-svhm.ipynb)
- Application to DCASE acoustic scene classification in [`grl_dcase.ipynb`](notebooks/grl_dcase.ipynb)
- Neural network architectures with feature extractors, classifiers, and discriminators
- Based on the paper: [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1409.7495)

### 2. PaSST Feature Extractor
- Practice implementation of pre-trained PaSST model in [`passt_pretrained_practise.ipynb`](notebooks/passt_pretrained_practise.ipynb)
- Audio feature extraction for acoustic scene classification
- Integration with domain adaptation frameworks

## Datasets Used

### Computer Vision
- **MNIST**: Handwritten digit recognition (downloaded automatically)
- **SVHN**: Street View House Numbers (downloaded automatically)

### Audio/Acoustic
- **DCASE TAU**: Urban acoustic scenes from multiple cities
  - Park scenes (Helsinki, Lisbon, Milan, Paris, Prague, Stockholm, Vienna)
  - Street pedestrian scenes 
  - Metro station scenes
  - Public square scenes
  - Bus scenes
- **Custom Audio Files**: Any .wav files can be used for PaSST feature extraction practice

## Dataset Setup

### Automatic Downloads
The MNIST and SVHN datasets will be automatically downloaded when running the respective notebooks for the first time.

### Manual Downloads Required
Due to file size limitations, the following datasets need to be downloaded separately:

1. **DCASE TAU Dataset**: 
   - Download from [DCASE Challenge website](http://dcase.community/challenge2020/task-acoustic-scene-classification)
   - Place audio files in `datasets/dcase/audio/`

2. **Audio Files for PaSST Practice**:
   - Any .wav audio files can be used
   - Place sample files in `datasets/audio_files/`
   - The notebook includes examples with files like:
     - `33711__acclivity__excessiveexposure.wav`
     - `222993__zyrytsounds__people-talking.wav`
     - `36105__erh__roswell.wav`

## Key Files

- [`grl_mnist-svhm.ipynb`](notebooks/grl_mnist-svhm.ipynb): Complete GRL implementation for visual domain adaptation
- [`grl_dcase.ipynb`](notebooks/grl_dcase.ipynb): GRL adapted for acoustic scene classification
- [`passt_pretrained_practise.ipynb`](notebooks/passt_pretrained_practise.ipynb): PaSST feature extractor exploration

## Requirements

The notebooks require the following Python packages:
- PyTorch
- torchvision
- librosa
- hear21passt
- numpy
- matplotlib

## Installation

```bash
# Install required packages
pip install torch torchvision librosa hear21passt numpy matplotlib

# Clone the repository
git clone https://github.com/RonnMath03/Unsupervised-Domain-Adaptation-Learning
cd Unsupervised-Domain-Adaptation-Learning

# Create datasets directory structure
mkdir -p datasets/dcase/audio
mkdir -p datasets/audio_files
```

## Usage

1. **MNIST-SVHN Domain Adaptation**:
   ```bash
   jupyter notebook notebooks/grl_mnist-svhm.ipynb
   ```

2. **DCASE Audio Domain Adaptation**:
   ```bash
   # First download DCASE dataset to datasets/dcase/audio/
   jupyter notebook notebooks/grl_dcase.ipynb
   ```

3. **PaSST Feature Extraction**:
   ```bash
   # Add any .wav files to datasets/audio_files/
   jupyter notebook notebooks/passt_pretrained_practise.ipynb
   ```

## Features

- **Automatic Data Handling**: MNIST and SVHN datasets are downloaded automatically
- **Flexible Audio Input**: PaSST implementation works with any .wav audio files
- **Batch Processing**: Support for both single audio file and batch audio processing
- **GPU Support**: CUDA-enabled implementations for faster processing

## References

### Research Paper
- Ganin, Y., & Lempitsky, V. (2015). [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495). *arXiv preprint arXiv:1409.7495*.

### Source Repositories
- **DANN PyTorch Implementation**: [Yangyangii/DANN-pytorch](https://github.com/Yangyangii/DANN-pytorch)
  - Used as base for GRL/DANN implementations in `grl_mnist-svhm.ipynb` and `grl_dcase.ipynb`
- **PaSST Implementation**: [kkoutini/PaSST](https://github.com/kkoutini/PaSST)
  - Used for audio feature extraction in `passt_pretrained_practise.ipynb`

## Acknowledgments

- **Code Attribution**: The implementations are adapted and modified from the above-mentioned GitHub repositories
- **Research Foundation**: Based on the seminal work by Ganin & Lempitsky on Domain-Adversarial Training
- **Institution**: This work is conducted as part of our summer internship preparation at IIT Mandi
- **Datasets**: 
  - DCASE dataset provided by the Detection and Classification of Acoustic Scenes and Events community
  - Audio samples from FreeSound community for PaSST practice

## Future Work

This repository serves as groundwork for our main project: **Domain Adaptation for Acoustic Scene Classification for different devices using Gradual Vanishing Bridge** at IIT Mandi.

## License

This project is for educational and research purposes. Please refer to the original source repositories for their respective licensing information:
- [DANN-pytorch License](https://github.com/Yangyangii/DANN-pytorch/blob/master/LICENSE)
- [PaSST License](https://github.com/kkoutini/PaSST/blob/main/LICENSE)

## Notes

- The `datasets/` folder structure exists but files need to be downloaded separately due to size constraints
- For PaSST practice, any .wav audio files can be used - no specific dataset required
- All notebooks include necessary package installation cells
- Code modifications were made to adapt the original implementations for our specific learning objectives
