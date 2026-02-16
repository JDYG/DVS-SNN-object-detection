# DVS-SNN Object Detection

Object Detection with Spiking Neural Networks on Event Videos for Driving Scenarios

## Overview

This project implements object detection on event camera data (DVS) using spiking neural networks. It provides two training approaches:

- **BPTT (Backpropagation Through Time (BPTT))**: Traditional temporal sequence training
- **TFBPTT (Truncated Final-step BPTT (TFBPTT))**: GPU Memory-efficient variant for longer sequences

## Key Features

- Event-based object detection using DVS sensors
- Spiking neural network architecture with Leaky Integrate-and-Fire (LIF) neurons
- Attention mechanisms for enhanced feature extraction
- YOLO-style detection heads
- Latent memory modules for temporal processing
- Support for both training and inference modes

## Architecture

The main model (`Gen1Spiking`) consists of:

1. **Embedding Module**: Processes raw event sequences
2. **Attention Module**: Captures spatial-temporal dependencies
3. **Latent Memory**: Maintains temporal state information
4. **Detection Head**: YOLO-based multi-scale object detection



## Directory Structure

```
├── DVS_SNN_GEN1_BPTT/
│   ├── GEN1_od/
│   │   ├── models_spiking/     # Model definitions
│   │   │   ├── model_single_step.py
│   │   │   ├── yolo.py         # YOLO detection head
│   │   │   ├── neurons.py      # Spiking neuron implementations
│   │   │   ├── attention.py
│   │   │   ├── blocks.py
│   │   │   └── builder.py
│   │   └── video_infer/        # Inference utilities
│   └── utils/                  # Utility functions
├── DVS_SNN_GEN1_TFBPTT/
│   └── (similar structure for truncated BPTT)
└── README.md
```

## Dependencies

- PyTorch 2.1.0
- SpikingJelly 0.0.0.0.15
- einops 0.7.0
- NumPy 1.24.3
- pytorch lightning 2.1

## Usage

### Training

```python
# Example training script
python DVS_SNN_GEN1_BPTT/GEN1_od/models_spiking/run_train_lab.py
```

### Inference

```python
# Example inference script
python DVS_SNN_GEN1_BPTT/GEN1_od/video_infer/run_infer.py
```

## Configuration

Model configurations are located in `config_test/` directories. Key parameters include:

- Input event dimensions
- Network architecture settings
- Detection thresholds
- Training hyperparameters

## Data Format

The files names and corresponding labels should be extracted to GEN1_od/data/xx.txt

# Acknowledgement


## License

This project is for research purposes.


