# Unsupervised Domain Adaptation Learning

This repository contains implementations of unsupervised domain adaptation techniques using Gradient Reversal Layer (GRL) and PaSST feature extractors across various datasets. The code was collected and modified from various GitHub sources as a learning exercise and precursor to our main research project.

## Project Overview

This work serves as foundational research for our summer internship project at **IIT Mandi**: *Domain Adaptation for Acoustic Scene Classification for different devices using Cyclic Self Training*.

**Status**: Practice work completed. Repository is finalized with no further updates planned.

## Repository Structure

```
├── notebooks/
│   ├── grl_cv/                        # GRL implementations for computer vision
│   ├── grl_dcase/                     # GRL implementations for DCASE dataset
│   └── passt/                         # PaSST feature extractor implementations
│       └── passt_pretrained_practise.ipynb # PaSST practice notebook
├── datasets/
│   ├── audio_files/                   # Sample audio files for PaSST practice
│   │   ├── 222993__zyrytsounds__people-talking.wav
│   │   ├── 33711__acclivity__excessiveexposure.wav
│   │   └── 36105__erh__roswell.wav
│   ├── cv_data/                       # Computer vision datasets (auto-downloaded)
│   └── dcase/                         # DCASE TAU dataset
│       ├── meta.csv                   # Dataset metadata
│       ├── README.md                  # DCASE dataset documentation
│       ├── README.html                # DCASE dataset documentation (HTML)
│       ├── test/                      # Test data organization
│       └── train/                     # Training data organization
│           ├── source/                # Source domain data (Device A)
│           └── target/                # Target domain data (Devices B,C,S1-S3)
├── .ipynb_checkpoints/                # Jupyter checkpoint files
├── .gitattributes
├── .gitignore
├── README.md                          # This file
└── structure.txt                      # Repository structure documentation
```

## Implemented Techniques

### 1. Gradient Reversal Layer (GRL/DANN)
- Computer vision implementations in [`grl_cv/`](notebooks/grl_cv/) folder
- DCASE audio implementations in [`grl_dcase/`](notebooks/grl_dcase/) folder
- Neural network architectures with feature extractors, classifiers, and discriminators
- Based on the paper: [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1409.7495)

### 2. PaSST Feature Extractor
- Practice implementation of pre-trained PaSST model in [`passt_pretrained_practise.ipynb`](notebooks/passt/passt_pretrained_practise.ipynb)
- Audio feature extraction for acoustic scene classification
- Integration with domain adaptation frameworks

## Datasets Used

### Computer Vision
- **MNIST**: Handwritten digit recognition (downloaded automatically)
- **SVHN**: Street View House Numbers (downloaded automatically)

### Audio/Acoustic
- **DCASE TAU 2020 Mobile**: Urban acoustic scenes from multiple cities and devices
  - 10 acoustic scene classes: airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram
  - Multiple devices: A (source), B, C, S1-S6 (targets)
  - 64 hours of audio data total
  - Detailed statistics in [`datasets/dcase/README.md`](datasets/dcase/README.md)
- **Custom Audio Files**: Sample .wav files for PaSST feature extraction practice

## Dataset Setup

### Automatic Downloads
The MNIST and SVHN datasets are automatically downloaded to [`datasets/cv_data/`](datasets/cv_data/) when running the respective notebooks.

### Manual Downloads Required
Due to file size limitations, the DCASE dataset needs to be downloaded separately:

1. **DCASE TAU 2020 Mobile Dataset**: 
   - Download from [DCASE Challenge website](http://dcase.community/challenge2020/task-acoustic-scene-classification)
   - Audio files should be placed in `datasets/dcase/audio/` (not included in repository)
   - Pre-organized train/test split folders available in [`datasets/dcase/train/`](datasets/dcase/train/) and [`datasets/dcase/test/`](datasets/dcase/test/)

2. **Audio Files for PaSST Practice**:
   - Sample files already included in [`datasets/audio_files/`](datasets/audio_files/)
   - Any additional .wav files can be added for experimentation

## Key Features

- **Domain Adaptation Focus**: Implementations specifically designed for cross-device acoustic scene classification
- **Complete Workflow**: From data loading to model training and evaluation
- **Flexible Audio Input**: PaSST implementation works with any .wav audio files
- **GPU Support**: CUDA-enabled implementations for faster processing
- **Organized Structure**: Clear separation between different techniques and datasets
- **Documentation**: Comprehensive dataset documentation and usage examples

## Technical Requirements

- Python 3.12+
- PyTorch
- torchvision
- librosa
- hear21passt
- numpy
- matplotlib
- scikit-learn

## Installation

```bash
# Clone the repository
git clone https://github.com/RonnMath03/Unsupervised-Domain-Adaptation-Learning
cd Unsupervised-Domain-Adaptation-Learning

# Install required packages
pip install torch torchvision librosa hear21passt numpy matplotlib scikit-learn

# Dataset directories already exist, download DCASE data if needed
```

## Usage

### PaSST Feature Extraction
```bash
# Practice with sample audio files
jupyter notebook notebooks/passt/passt_pretrained_practise.ipynb
```

### GRL/DANN Implementations
```bash
# Computer vision domain adaptation
jupyter notebook notebooks/grl_cv/[specific_notebook].ipynb

# Audio domain adaptation
jupyter notebook notebooks/grl_dcase/[specific_notebook].ipynb
```

## References

### Research Papers
- Ganin, Y., & Lempitsky, V. (2015). [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495). *arXiv preprint arXiv:1409.7495*.
- Koutini, K., et al. (2021). PaSST: Efficient Training of Audio Transformers with Patchout.

### Source Repositories
- **DANN PyTorch Implementation**: [Yangyangii/DANN-pytorch](https://github.com/Yangyangii/DANN-pytorch)
  - Used as base for GRL/DANN implementations
- **PaSST Implementation**: [kkoutini/PaSST](https://github.com/kkoutini/PaSST)
  - Used for audio feature extraction

### Datasets
- **DCASE TAU 2020 Mobile**: [Detection and Classification of Acoustic Scenes and Events](http://dcase.community/challenge2020/task-acoustic-scene-classification)
- **Audio Samples**: [FreeSound Community](https://freesound.org/)

## Acknowledgments

- **Code Attribution**: Implementations adapted and modified from open-source GitHub repositories
- **Research Foundation**: Based on Domain-Adversarial Training and PaSST architectures
- **Institution**: Summer internship preparation work at IIT Mandi
- **Dataset Providers**: 
  - DCASE community for acoustic scene datasets
  - Tampere University for TAU Urban Acoustic Scenes dataset
  - FreeSound community for sample audio files

## Future Work

This repository serves as groundwork for our main research project: **Domain Adaptation for Acoustic Scene Classification for different devices using Cyclic Self Training** at IIT Mandi.

The techniques and implementations explored here will inform:
- Advanced domain adaptation strategies
- Multi-device acoustic scene classification
- Gradual domain shift mitigation techniques

## License

This project is for educational and research purposes. Please refer to the original source repositories for their respective licensing information:
- [DANN-pytorch License](https://github.com/Yangyangii/DANN-pytorch/blob/master/LICENSE)
- [PaSST License](https://github.com/kkoutini/PaSST/blob/main/LICENSE)
- DCASE TAU dataset: Academic use only, commercial use prohibited

## Final Notes

- **Completion Status**: All practice implementations are complete and finalized
- **Repository Maintenance**: No further updates planned for this practice repository
- **Dataset Structure**: Organized for easy domain adaptation experiments
- **Code Quality**: All notebooks include proper documentation and usage examples
- **Learning Outcome**: Successfully explored UDA techniques for acoustic scene classification
- **Next Steps**: Proceed to main research project implementation

---

**Repository Status**: **COMPLETED**
