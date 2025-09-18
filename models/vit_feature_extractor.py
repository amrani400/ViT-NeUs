import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .dinov2 import DINOv2

class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_size='vitl', patch_size=14, layer_to_extract=23):
        super(ViTFeatureExtractor, self).__init__()
        self.patch_size = patch_size
        self.layer_to_extract = layer_to_extract
        self.vit = DINOv2(model_name=vit_size, patch_size=patch_size)
        self.embed_dim = self.vit.embed_dim  # 1024 for vitl
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, images):
        images = self.preprocess(images).to(next(self.vit.parameters()).device)
      
        B, C, H, W = images.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom))
        features = self.vit.get_intermediate_layers(images, n=[23], reshape=True, norm=True)[0]  
        padded_size = (images.shape[2], images.shape[3])

      

        
        low = F.avg_pool2d(features, kernel_size=2, stride=2)  
        up_low = F.interpolate(low, size=features.shape[2:], mode='bilinear', align_corners=True)
        high = features - up_low  

        
        low = low.squeeze(0)
        high = high.squeeze(0)

      

        return low, high, padded_size  