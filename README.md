# CT<->MRI Image Translation Using ResNet and UNet

This project implements bidirectional translation between CT (Computed Tomography) and MRI (Magnetic Resonance Imaging) medical images using CycleGAN with ResNet and UNet architectures.

## Overview

The repository contains implementations of two deep learning architectures for medical image translation:

- ResNet-based CycleGAN
- UNet-based CycleGAN 

Both models are trained to perform:
- CT → MRI translation
- MRI → CT translation

## Sample Results

### CT to MRI Translation (ResNet)
<p align="center">
<img src="resnet/resnet_ct_mri/sample gen/real_img_gen_img_1.png" width="800">
<br>
<em>Left: Original CT scan, Right: Generated MRI-like image</em>
</p>

### MRI to CT Translation (ResNet) 
<p align="center">
<img src="resnet/resnet_mri_ct/sample gen/real_img_gen_img_1.png" width="800">
<br>
<em>Left: Original MRI scan, Right: Generated CT-like image</em>
</p>

### CT to MRI Translation (UNet)
<p align="center">
<img src="unet/unet_ct_mri/sample gen/real_img_gen_img_1.png" width="800">
<br>
<em>Left: Original CT scan, Right: Generated MRI-like image</em>
</p>

### MRI to CT Translation (UNet)
<p align="center">
<img src="unet/unet_mri_ct/sample gen/real_img_gen_img_1.png" width="800">
<br>
<em>Left: Original MRI scan, Right: Generated CT-like image</em>
</p>

## Project Structure

```
├── resnet/
│   ├── resnet_ct_mri/         # ResNet model for CT to MRI
│   │   ├── sample gen/        # Generated sample images
│   │   └── checkpoints/       # Model checkpoints
│   ├── resnet_mri_ct/         # ResNet model for MRI to CT 
│   │   ├── sample gen/        # Generated sample image
│   │   └── checkpoints/       # Model checkpoints
│   ├── resnet-ct-mri.ipynb    # Training notebook for CT to MRI
│   └── resnet-mri-ct.ipynb    # Training notebook for MRI to CT
│
└── unet/
    ├── unet_ct_mri/           # UNet model for CT to MRI 
    │   ├── sample gen/        # Generated sample images
    │   └── checkpoints/       # Model checkpoints
    ├── unet_mri_ct/           # UNet model for MRI to CT 
    │   ├── sample gen/        # Generated sample images
    │   └── checkpoints/       # Model checkpoints
    ├── unet-ct-mri.ipynb      # Training notebook for CT to MRI
    └── unet-mri-ct.ipynb      # Training notebook for MRI to CT
```

## Requirements

```bash
pip install deepspeed
pip install lightning
pip install torchmetrics
pip install torch-fidelity
```

## Model Configuration

Key parameters can be modified in the notebooks:

```python
MODEL_CONFIG = {
    "gen_name": "resnet",    # Generator architecture: 'resnet' or 'unet'  
    "num_resblocks": 6,      # Number of residual blocks (ResNet only)
    "hid_channels": 64,      # Number of hidden channels
    "optimizer": ds.ops.adam.FusedAdam,  # Optimizer type
    "lr": 2e-4,             # Learning rate
    "betas": (0.5, 0.999),  # Adam optimizer betas
    "lambda_idt": 0.5,      # Identity loss weight
    "lambda_cycle": (10, 10), # Cycle consistency loss weights
    "buffer_size": 100,     # Image buffer size
    "num_epochs": 120,      # Number of training epochs
    "decay_epochs": 10      # Learning rate decay start epoch
}
```

## Training

To train a model:

1. Open the desired notebook (e.g., `resnet-ct-mri.ipynb`)
2. Configure training parameters in `MODEL_CONFIG` 
3. Configure data paths in `DM_CONFIG`
4. Run all cells

The training progress will show:
- Generator loss
- Discriminator losses (MRI and CT)
- Current learning rate
- Sample translations every N epochs

## Evaluation

Models are evaluated using:
- Visual comparison of generated images
- Cycle-consistency loss
- Identity mapping loss
- Adversarial loss metrics

## Inference

To generate translations using a trained model:

```python
# Load model
model = CycleGAN.load_from_checkpoint("path/to/checkpoint.ckpt")

# Generate MRI from CT
mri = model(ct_image) 

# Generate CT from MRI  
ct = model.gen_MRI_CT(mri_image)
```

## Acknowledgements

This implementation uses:
- [PyTorch Lightning](https://lightning.ai/)
- [DeepSpeed](https://www.deepspeed.ai/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/)
- [CycleGAN](https://junyanz.github.io/CycleGAN/) architecture

## License

This project is licensed under the MIT License - see the LICENSE file for details.
