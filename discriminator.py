from layer import ConvBlock
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    """Improved Discriminator with PatchGAN architecture"""
    def __init__(self, category_num: int, img_dim: int = 2, disc_dim: int = 64):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(img_dim, disc_dim, use_bn=False),
            ConvBlock(disc_dim, disc_dim*2),
            ConvBlock(disc_dim*2, disc_dim*4),
            ConvBlock(disc_dim*4, disc_dim*8)
        )
        
        # PatchGAN output
        self.patch_out = nn.Conv2d(disc_dim*8, 1, kernel_size=4, stride=1, padding=1)
        
        # Category classification
        self.category_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(disc_dim*8, category_num)
        )
        
    def forward(self, x):
        features = self.conv_blocks(x)
        patch_score = self.patch_out(features)
        category_score = self.category_out(features)
        
        return torch.sigmoid(patch_score), patch_score, category_score
