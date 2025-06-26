# UDA Gradient Reversal Layer

This repository contains the first internship project completed during the research internship at IIT Mandi implementing unsupervised domain adaptation techniques using Gradient Reversal Layer (GRL) and PaSST feature extractors. We started by re-implementing and validating existing GRL approaches on computer vision datasets, then extended our work to audio domain adaptation using PaSST feature extractors on the DCASE dataset - representing our novel contribution to cross-device acoustic scene classification.

## Repository Structure

```
├── grl_cv/                           # GRL implementations for computer vision (validation work)
│   ├── baseline_wo_grl_svhn.ipynb    # Baseline without GRL: SVHN→MNIST
│   ├── dann_svhn_w_grl.ipynb         # DANN with GRL: SVHN→MNIST
│   └── cv_data/                      # Computer vision datasets (auto-downloaded)
│       ├── test_32x32.mat           # SVHN test data
│       ├── train_32x32.mat          # SVHN train data
│       └── MNIST/                    # MNIST dataset
├── grl_dcase/                        # GRL implementations for DCASE dataset (our main work)
│   ├── dann_dcase_w_grl.ipynb        # DANN with GRL for DCASE TAU 2020
│   ├── baseline_dcase_wo_ds.ipynb    # Baseline without domain shift for DCASE TAU 2020
│   ├── source_only_dcase_w_ds.ipynb  # Source-only baseline with domain shift for DCASE TAU 2020
│   ├── dcase/                        # DCASE TAU dataset
│   │   ├── meta.csv                  # Dataset metadata
│   │   ├── README.md                 # DCASE dataset documentation
│   │   ├── README.html               # DCASE dataset documentation (HTML)
│   │   └── evaluation_setup/         # Cross-validation setup for data splits
│   └── saved_models/                 # Trained model checkpoints
├── passt/                            # PaSST feature extractor implementations
│   ├── passt_dcase.ipynb             # PaSST with DCASE dataset
│   ├── passt_pretrained_practise.ipynb # PaSST practice notebook
│   └── audio_files/                  # Sample audio files for PaSST practice
├── .ipynb_checkpoints/               # Jupyter checkpoint files
├── .gitattributes
├── .gitignore
└── README.md                         # This file
```

## Project Overview

### Phase 1: Validation and Understanding (Computer Vision)
We began by re-implementing existing Domain-Adversarial Neural Network (DANN) approaches from GitHub repositories to:
- Validate the correctness of existing implementations
- Understand the theoretical foundations of GRL-based domain adaptation
- Establish baseline performance metrics on standard CV datasets

### Phase 2: Audio Application (Audio Domain Adaptation)
Building on our understanding, we developed implementations for acoustic scene classification:
- Integrated PaSST (pre-trained audio transformers) as feature extractors
- Adapted GRL techniques for cross-device audio domain adaptation
- Implemented comprehensive evaluation on DCASE TAU 2020 dataset
- Source/target split for both training and testing data

## Implemented Techniques

### 1. Computer Vision Domain Adaptation (Validation Work)

#### SVHN → MNIST Domain Adaptation
- **[`baseline_wo_grl_svhn.ipynb`](grl_cv/baseline_wo_grl_svhn.ipynb)**: Baseline implementation without GRL
  - Source domain: SVHN (Street View House Numbers)
  - Target domain: MNIST (Handwritten digits)
  - CNN feature extractor + classifier architecture
  - **Maximum Target Accuracy**: 63.1%

- **[`dann_svhn_w_grl.ipynb`](grl_cv/dann_svhn_w_grl.ipynb)**: DANN implementation with GRL
  - Domain-adversarial training with gradient reversal layer
  - CNN feature extractor + classifier + discriminator architecture
  - **Maximum Target Accuracy**: 72.1%
  - **Improvement**: +9% over baseline, demonstrating GRL effectiveness

### 2. Audio Domain Adaptation (Main Contribution)

#### Cross-Device Acoustic Scene Classification
- **[`dann_dcase_w_grl.ipynb`](grl_dcase/dann_dcase_w_grl.ipynb)**: DANN with GRL for acoustic scenes
  - **Novel Integration**: PaSST feature extractor with GRL framework
  - Multi-device domain adaptation (Device A → Devices B,C,S1-S6)
  - 10 acoustic scene classes from DCASE TAU 2020


- **[`baseline_dcase_wo_ds.ipynb`](grl_dcase/baseline_dcase_wo_ds.ipynb)**: Baseline without domain shift
  - No domain adaptation (all devices used for training)
  - Same PaSST backbone for fair comparison
  - Upper bound performance reference

- **[`source_only_dcase_w_ds.ipynb`](grl_dcase/source_only_dcase_w_ds.ipynb)**: Source-only with domain shift
  - Training only on source domain (Device A)
  - Evaluation on target domains (Devices B,C,S1-S6)
  - Lower bound performance reference

### 3. PaSST Feature Extraction Framework
- **[`passt_pretrained_practise.ipynb`](passt/passt_pretrained_practise.ipynb)**: PaSST model exploration
  - Pre-trained AudioSet model integration
  - Audio preprocessing and feature extraction pipelines
  - Foundation for DCASE implementations

- **[`passt_dcase.ipynb`](passt/passt_dcase.ipynb)**: PaSST-DCASE integration
  - DCASE-specific preprocessing
  - Feature extraction optimization for domain adaptation
  - Performance analysis across devices

## Data Split Strategy

Evaluation protocol for the DCASE dataset:

### Training Split
- **Source Domain**: Device A recordings (primary recording device)
- **Target Domain**: Devices B,C,S1-S3 recordings (secondary devices)

### Testing Split  
- **Source Test**: Device A test recordings
- **Target Test**: Devices B,C,S1-S6 test recordings (including additional unseen devices S4-S6)

This split allows comprehensive evaluation of domain adaptation across varying device characteristics and recording conditions.

## Datasets

### Computer Vision (Validation)
- **MNIST**: Handwritten digits (60k train, 10k test)
- **SVHN**: Street View House Numbers (73k train, 26k test)
- **Auto-download**: Handled by torchvision datasets

### Audio (Main Work)
- **DCASE TAU 2020 Mobile**: Urban acoustic scenes
  - **Source**: Device A (10,215 training files)
  - **Target**: Devices B,C,S1-S6 (3,747 with varying counts per device)
  - **Classes**: 10 acoustic scenes (airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram)
  - **Total**: 64 hours of audio data
  - **Format**: 10-second segments, 32kHz, mono

## Results and Performance

### Computer Vision Domain Adaptation (SVHN→MNIST)

| Method | Source Accuracy | Target Accuracy | Improvement |
|--------|----------------|-----------------|-------------|
| Baseline w/o GRL | 93.4% | 63.1% | -T |
| DANN w/ GRL | 93.7% | 72.1% | **+9.0%** |

### Audio Domain Adaptation (DCASE TAU 2020)

| Method                     | Source Accuracy | Target Accuracy      | Overall Accuracy | Performance Notes               |
|----------------------------|---------------  |----------------------|------------------|---------------------------------|
| Baseline w/o Domain Shift  |     77.27%      |      71.08%          |     74.17%       | Upper bound (no domain gap)     |
| **DANN w/ GRL**            |     76.67%      |      64.06%          |   **70.37%**     | **Our main result**             |
| Source-only w/ Domain Shift|     81.21%      |      51.86%          |     47.61%       | Lower bound (with domain gap)   |


**Key Achievement**: Our DANN implementation with PaSST features achieves 70.37%% accuracy, representing a **22.76% improvement** over source-only training and performing within 3.8% of the no-domain-shift upper bound.

## Technical Implementation

### Contributions
1. **PaSST-GRL Integration**: Implementation combining pre-trained audio transformers with gradient reversal layers
2. **DCASE Evaluation**: Source/target split strategy for comprehensive cross-device evaluation in both train and test
3. **Audio Domain Adaptation Pipeline**: End-to-end framework for acoustic scene classification with domain adaptation

### Architecture Details
- **Feature Extractor**: PaSST (pre-trained on AudioSet, 768-dim features)
- **Adaptation Layers**: 768→512→256 with batch normalization and dropout
- **Classifier**: 256→10 classes for acoustic scenes
- **Discriminator**: 256→2 for domain classification
- **GRL**: Gradient reversal with adaptive λ scheduling

## Setup and Usage

### Installation
```bash
# Clone the repository
git clone https://github.com/UDA-IIT-Mandi/UDA-Gradient-Reversal-Layer.git
cd UDA-Gradient-Reversal-Layer

# Install required packages
pip install torch torchvision librosa hear21passt numpy matplotlib scikit-learn pandas
```

### Running Experiments

#### Computer Vision (Validation)
```bash
# Baseline without GRL
jupyter notebook grl_cv/baseline_wo_grl_svhn.ipynb

# DANN with GRL
jupyter notebook grl_cv/dann_svhn_w_grl.ipynb
```

#### Audio Domain Adaptation (Main Work)
```bash
# DANN implementation with PaSST
jupyter notebook grl_dcase/dann_dcase_w_grl.ipynb

# Baseline comparisons
jupyter notebook grl_dcase/baseline_dcase_wo_ds.ipynb
jupyter notebook grl_dcase/source_only_dcase_w_ds.ipynb
```

#### PaSST Exploration
```bash
# PaSST practice and exploration
jupyter notebook passt/passt_pretrained_practise.ipynb
jupyter notebook passt/passt_dcase.ipynb
```

### Dataset Preparation
- **CV Datasets**: Automatically downloaded via torchvision
- **DCASE Dataset**: Download from [DCASE Challenge](http://dcase.community/challenge2020/task-acoustic-scene-classification)
  - Place audio files in `grl_dcase/dcase/audio/`
  - Metadata and evaluation setup already included

## Code Attribution and Acknowledgments

### Original Implementations (Validation Work)
- **DANN PyTorch**: Based on [Yangyangii/DANN-pytorch](https://github.com/Yangyangii/DANN-pytorch)
  - Original GRL and DANN implementations for computer vision
  - We re-implemented and validated their approach on SVHN→MNIST
- **PaSST**: Based on [kkoutini/PaSST](https://github.com/kkoutini/PaSST)
  - Pre-trained audio transformer models
  - We integrated PaSST as feature extractors in our domain adaptation framework

### Research Foundation
- Ganin, Y., & Lempitsky, V. (2015). [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495)
- Koutini, K., et al. (2021). [Efficient Training of Audio Transformers with Patchout](https://arxiv.org/abs/2110.05069)

### Dataset Acknowledgments
- **DCASE TAU 2020**: Detection and Classification of
Acoustic Scenes and Events, Tampere University, Audio Research Group
- **FreeSound**: Community-contributed audio samples

## Future Directions

This project establishes the foundation for advanced domain adaptation research in acoustic scene classification:

1. **Cycle Self-Training**: Extending DANN with cycle-self-training
2. **Progressive Domain Adaptation**: Gradual adaptation across multiple device types
3. **Multi-Source Domain Adaptation**: Leveraging multiple source domains
4. **Attention-Based Adaptation**: Incorporating attention mechanisms in domain adaptation

## License

This project is for academic and research purposes. Please refer to original repositories for licensing:
- [DANN-pytorch License](https://github.com/Yangyangii/DANN-pytorch/blob/master/LICENSE)
- [PaSST License](https://github.com/kkoutini/PaSST/blob/main/LICENSE)
- DCASE TAU dataset: Academic use, commercial use prohibited

---

**Project Status**: Completed
