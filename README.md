# Unsupervised Domain Adaptation Learning

This repository contains implementations of unsupervised domain adaptation techniques using Gradient Reversal Layer (GRL) and PaSST feature extractors across various datasets. The code was collected and modified from various GitHub sources as a learning exercise and precursor to our main research project.

## Project Overview

This work serves as foundational research for our summer internship project at **IIT Mandi**: *Domain Adaptation for Acoustic Scene Classification for different devices using Cyclic Self Training*.

**Status**: Practice work completed. Repository is finalized with no further updates planned.

## Repository Structure

```
├── notebooks/
│   ├── grl_cv/                        # GRL implementations for computer vision
│   │   ├── baseline_no_grl.ipynb     # Baseline without GRL: SVHN→MNIST
│   │   └── dann_with_grl.ipynb       # DANN with GRL: SVHN→MNIST
│   ├── grl_dcase/                     # GRL implementations for DCASE dataset
│   │   ├── dann_dcase.ipynb          # DANN with GRL for DCASE TAU 2020
│   │   ├── baseline_dcase_wo_da.ipynb # Baseline without domain shift for DCASE TAU 2020
│   │   └── source_only_dcase_w_da.ipynb # Baseline without GRL but with domain adpatation for DCASE TAU 2020
│   └── passt/                         # PaSST feature extractor implementations
│       ├── passt_pretrained_practise.ipynb # PaSST practice notebook
│       └── passt_dcase.ipynb         # PaSST with DCASE dataset
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
│       ├── evaluation_setup/          # Cross-validation setup for data splits
│       │   ├── fold1_train.csv        # Training file list (used for train/source & train/target splits)
│       │   ├── fold1_test.csv         # Testing file list (used for test split)
│       │   └── fold1_evaluate.csv     # Evaluation file list with ground truth labels
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

#### Computer Vision Domain Adaptation
- **[`baseline_no_grl.ipynb`](notebooks/grl_cv/baseline_no_grl.ipynb)**: Baseline implementation without GRL for SVHN→MNIST domain adaptation
  - Source domain: SVHN (Street View House Numbers)
  - Target domain: MNIST (Handwritten digits)
  - Feature extractor + classifier architecture
  - Performance comparison baseline for GRL methods

- **[`dann_with_grl.ipynb`](notebooks/grl_cv/dann_with_grl.ipynb)**: Full DANN implementation with GRL for SVHN→MNIST domain adaptation
  - Domain-adversarial training with gradient reversal layer
  - CNN feature extractor + classifier + discriminator architecture
  - Comparison with baseline to demonstrate GRL effectiveness

#### Audio Domain Adaptation (DCASE TAU 2020)
- **[`dann_dcase.ipynb`](notebooks/grl_dcase/dann_dcase.ipynb)**: Full DANN implementation with GRL for acoustic scene classification
  - Domain-adversarial training with gradient reversal layer
  - PaSST feature extractor (pre-trained on AudioSet)
  - Multi-device domain adaptation (Device A → Devices B,C,S1-S6)
  - 10 acoustic scene classes
  - Confusion matrix analysis and device-specific performance evaluation

- **[`baseline_dcase_wo_da.ipynb`](notebooks/grl_dcase/baseline_dcase_wo_da.ipynb)**: Baseline implementation without GRL and Domain Shift for DCASE dataset
  - Source and Target (fully labelled without domain shift) training for comparison with DANN
  - Same PaSST backbone as DANN implementation
  - Performance baseline for comparing domain adaptation against models with no domain shift
 
- **[`source_only_dcase_w_da.ipynb`](notebooks/grl_dcase/source_only_dcase_w_da.ipynb)**: Baseline implementation without GRL for DCASE dataset
  - Source-only training for comparison with DANN
  - Same PaSST backbone as DANN implementation
  - Performance baseline for measuring domain adaptation improvements over models with domain shift but no GRL

### 2. PaSST Feature Extractor
- **[`passt_pretrained_practise.ipynb`](notebooks/passt/passt_pretrained_practise.ipynb)**: Practice implementation of pre-trained PaSST model
  - Audio feature extraction for acoustic scene classification
  - Integration with domain adaptation frameworks
  - Sample audio file processing examples

- **[`passt_dcase.ipynb`](notebooks/passt/passt_dcase.ipynb)**: PaSST feature extraction specifically for DCASE dataset
  - DCASE TAU 2020 dataset preprocessing and feature extraction
  - PaSST model integration with acoustic scene classification
  - Preparation for domain adaptation experiments

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

## DCASE Dataset Organization

The DCASE TAU Urban Acoustic Scenes 2020 Mobile dataset uses a specific cross-validation setup provided in the [`datasets/dcase/evaluation_setup/`](datasets/dcase/evaluation_setup/) folder:

### Data Split Files
- **[`fold1_train.csv`](datasets/dcase/evaluation_setup/fold1_train.csv)**: Contains the training file list with scene labels
  - Used to split audio files into source domain (Device A) and target domain (Devices B,C,S1-S3)
  - Format: `[audio file][tab][scene label]`
  - Device A files → [`datasets/dcase/train/source/`](datasets/dcase/train/source/)
  - Device B,C,S1-S3 files → [`datasets/dcase/train/target/`](datasets/dcase/train/target/)

- **[`fold1_test.csv`](datasets/dcase/evaluation_setup/fold1_test.csv)**: Contains the testing file list
  - Used for evaluation across all devices (A,B,C,S1-S6)
  - Format: `[audio file]`
  - All test files → [`datasets/dcase/test/`](datasets/dcase/test/)

- **[`fold1_evaluate.csv`](datasets/dcase/evaluation_setup/fold1_evaluate.csv)**: Same as test list but with ground truth labels
  - Used for final evaluation and performance metrics
  - Format: `[audio file][tab][scene label]`

### Domain Adaptation Setup
- **Source Domain**: Device A recordings (10,215 files)
- **Target Domain**: Devices B,C,S1-S3 recordings (3,747 files) 
- **Test Set**: All devices A,B,C,S1-S6 recordings (2,968 files)
- **Scene Classes**: 10 acoustic scenes (airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram)

The data split ensures that segments recorded at the same location are kept in the same subset to prevent data leakage between training and testing.

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

### Computer Vision Domain Adaptation
```bash
# Baseline without GRL (SVHN→MNIST)
jupyter notebook notebooks/grl_cv/baseline_no_grl.ipynb

# DANN with GRL (SVHN→MNIST)
jupyter notebook notebooks/grl_cv/dann_with_grl.ipynb
```

### Audio Domain Adaptation (DCASE)
```bash
# DANN with GRL implementation
jupyter notebook notebooks/grl_dcase/dann_dcase.ipynb

# Baseline without GRL
jupyter notebook notebooks/grl_dcase/baseline_dcase.ipynb
```

### PaSST Feature Extraction
```bash
# Practice with sample audio files
jupyter notebook notebooks/passt/passt_pretrained_practise.ipynb

# PaSST with DCASE dataset
jupyter notebook notebooks/passt/passt_dcase.ipynb
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
