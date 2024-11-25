import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True, dropout=0.0):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, img_dim=1, conv_dim=64):
        super().__init__()
        
        # Encoder layers
        self.conv1 = ConvBlock(img_dim, conv_dim, use_bn=False)      # 128x128 -> 64x64
        self.conv2 = ConvBlock(conv_dim, conv_dim * 2)               # 64x64 -> 32x32
        self.conv3 = ConvBlock(conv_dim * 2, conv_dim * 4)          # 32x32 -> 16x16
        self.conv4 = ConvBlock(conv_dim * 4, conv_dim * 8)          # 16x16 -> 8x8
        self.conv5 = ConvBlock(conv_dim * 8, conv_dim * 8)          # 8x8 -> 4x4
        self.conv6 = ConvBlock(conv_dim * 8, conv_dim * 8)          # 4x4 -> 2x2
        
    def forward(self, x):
        skip_connections = {}
        
        # 각 레이어의 출력을 저장하고 shape 출력
        skip_connections['e1'] = x1 = self.conv1(x)
        # print(f"e1 shape: {x1.shape}")
        
        skip_connections['e2'] = x2 = self.conv2(x1)
        # print(f"e2 shape: {x2.shape}")
        
        skip_connections['e3'] = x3 = self.conv3(x2)
        # print(f"e3 shape: {x3.shape}")
        
        skip_connections['e4'] = x4 = self.conv4(x3)
        # print(f"e4 shape: {x4.shape}")
        
        skip_connections['e5'] = x5 = self.conv5(x4)
        # print(f"e5 shape: {x5.shape}")
        
        encoded = self.conv6(x5)
        # print(f"encoded shape: {encoded.shape}")
        skip_connections['e6'] = encoded
        
        return encoded, skip_connections

class Decoder(nn.Module):
    def __init__(self, img_dim=1, embedded_dim=640, conv_dim=64):
        super().__init__()
        
        # Decoder layers
        self.deconv1 = DeconvBlock(embedded_dim, conv_dim * 8, dropout=0.5)      # 2x2 -> 4x4
        self.deconv2 = DeconvBlock(conv_dim * 16, conv_dim * 8, dropout=0.5)     # 4x4 -> 8x8
        self.deconv3 = DeconvBlock(conv_dim * 16, conv_dim * 4)                  # 8x8 -> 16x16
        self.deconv4 = DeconvBlock(conv_dim * 8, conv_dim * 2)                   # 16x16 -> 32x32
        self.deconv5 = DeconvBlock(conv_dim * 4, conv_dim)                       # 32x32 -> 64x64
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 2, img_dim, 4, 2, 1),                  # 64x64 -> 128x128
            nn.Tanh()
        )
        
    def forward(self, x, skip_connections):
        # 각 레이어의 입출력 shape 출력
        # print(f"\nDecoder input shape: {x.shape}")
        
        x = self.deconv1(x)
        # print(f"After deconv1: {x.shape}")
        x = torch.cat([x, skip_connections['e5']], dim=1)
        # print(f"After skip connection 1: {x.shape}")
        
        x = self.deconv2(x)
        # print(f"After deconv2: {x.shape}")
        x = torch.cat([x, skip_connections['e4']], dim=1)
        # print(f"After skip connection 2: {x.shape}")
        
        x = self.deconv3(x)
        # print(f"After deconv3: {x.shape}")
        x = torch.cat([x, skip_connections['e3']], dim=1)
        # print(f"After skip connection 3: {x.shape}")
        
        x = self.deconv4(x)
        # print(f"After deconv4: {x.shape}")
        x = torch.cat([x, skip_connections['e2']], dim=1)
        # print(f"After skip connection 4: {x.shape}")
        
        x = self.deconv5(x)
        # print(f"After deconv5: {x.shape}")
        x = torch.cat([x, skip_connections['e1']], dim=1)
        # print(f"After skip connection 5: {x.shape}")
        
        x = self.deconv6(x)
        # print(f"Final output shape: {x.shape}")
        
        return x