
# ViT-NeUs
Code for anonymous 3D reconstruction submission.

## Installation
### Minimum Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 1.10+ with CUDA support (e.g., `torch>=1.10.0+cu113`)
- **NumPy**: 1.19+
- **PyHocon**: 0.3.5+
- **OpenCV-Python**: 4.5+
- **Trimesh**: 3.9+
- **TensorBoard**: 2.6+
- **CUDA**: 11.3+ (or compatible with your GPU)

### Dependencies
Install required packages via pip:
```bash
pip install torch torchvision numpy pyhocon opencv-python trimesh tensorboard
```

### train
```
python exp_runner.py --conf womask.conf --mode train --case CASE_NAME
```

### validate
```
python exp_runner.py --conf womask.conf --mode validate_mesh --mcube_threshold 0.0 --resolution 1024
```
## Dataset
the insect dataset is available at : https://drive.google.com/file/d/1QkLr9OU5NzqOv8t92-HjJTL6OKAbm9xg/view?usp=sharing
