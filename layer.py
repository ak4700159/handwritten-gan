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
    def __init__(self, img_dim=1, conv_dim=128):  # 기본 채널 수를 128로 증가
        super().__init__()
        
        self.conv1 = ConvBlock(img_dim, conv_dim, use_bn=False)      # 128x128 -> 64x64
        self.conv2 = ConvBlock(conv_dim, conv_dim * 2)               # 64x64 -> 32x32
        self.conv3 = ConvBlock(conv_dim * 2, conv_dim * 4)          # 32x32 -> 16x16
        self.conv4 = ConvBlock(conv_dim * 4, conv_dim * 8)          # 16x16 -> 8x8
        self.conv5 = ConvBlock(conv_dim * 8, conv_dim * 8)          # 8x8 -> 4x4
        
        # 4x4에서 3x3으로 변환하는 특별한 컨볼루션 레이어
        self.conv6 = nn.Sequential(
            nn.Conv2d(conv_dim * 8, conv_dim * 8, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 4x4 -> 3x3
        
        # 3x3 크기를 유지하면서 특징을 정제하는 레이어들
        self.conv7 = ConvBlock(conv_dim * 8, conv_dim * 8,
                             kernel_size=1, stride=1, padding=0)     # 3x3 유지
        self.conv8 = ConvBlock(conv_dim * 8, conv_dim * 8,
                             kernel_size=1, stride=1, padding=0)     # 3x3 유지     
        
    def forward(self, x):
        # 모든 중간 특징들을 저장하기 위한 딕셔너리
        skip_connections = {}
        
        # 각 레이어의 출력을 저장하여 디코더의 skip connection에서 사용
        skip_connections['e1'] = x1 = self.conv1(x)
        skip_connections['e2'] = x2 = self.conv2(x1)
        skip_connections['e3'] = x3 = self.conv3(x2)
        skip_connections['e4'] = x4 = self.conv4(x3)
        skip_connections['e5'] = x5 = self.conv5(x4)
        skip_connections['e6'] = x6 = self.conv6(x5)
        skip_connections['e7'] = x7 = self.conv7(x6)
        encoded = self.conv8(x7)
        skip_connections['e8'] = encoded
        
        return encoded, skip_connections

class Decoder(nn.Module):
    def __init__(self, img_dim=1, embedded_dim=1152, conv_dim=128):  # embedded_dim을 인코더 출력(1024) + 임베딩(128)으로 수정
        super().__init__()
        
        # 3x3 크기의 입력을 처리하는 첫 번째 레이어
        self.deconv1 = DeconvBlock(
            embedded_dim,
            conv_dim * 8,
            kernel_size=1,
            stride=1, 
            padding=0,
            dropout=0.5
        )  # 3x3 유지
        
        # 3x3 크기를 유지하는 두 번째 레이어
        self.deconv2 = DeconvBlock(
            conv_dim * 16,
            conv_dim * 8,
            kernel_size=1,     # 3x3 크기 유지
            stride=1,
            padding=0,
            dropout=0.5
        )  # 3x3 유지
        
        # 3x3에서 4x4로 변환하는 특별한 레이어
        self.deconv3 = DeconvBlock(
            conv_dim * 16,
            conv_dim * 8,
            kernel_size=2,     # 3x3 -> 4x4
            stride=1,
            padding=0,
            dropout=0.5
        )
        
        # 이후의 일반적인 디컨볼루션 레이어들
        self.deconv4 = DeconvBlock(conv_dim * 16, conv_dim * 8)    # 4x4 -> 8x8
        self.deconv5 = DeconvBlock(conv_dim * 16, conv_dim * 4)    # 8x8 -> 16x16
        self.deconv6 = DeconvBlock(conv_dim * 8, conv_dim * 2)     # 16x16 -> 32x32
        self.deconv7 = DeconvBlock(conv_dim * 4, conv_dim)         # 32x32 -> 64x64
        
        self.deconv8 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 2, img_dim, 4, 2, 1),
            nn.Tanh()
        )  # 64x64 -> 128x128

        
    def forward(self, x, skip_connections):
        # 각 단계별로 특징 맵의 크기를 출력하여 디버깅을 돕습니다
        x = self.deconv1(x)
        x = torch.cat([x, skip_connections['e7']], dim=1)
        
        x = self.deconv2(x)
        x = torch.cat([x, skip_connections['e6']], dim=1)
        
        x = self.deconv3(x)
        x = torch.cat([x, skip_connections['e5']], dim=1)
        
        x = self.deconv4(x)
        x = torch.cat([x, skip_connections['e4']], dim=1)
        
        x = self.deconv5(x)
        x = torch.cat([x, skip_connections['e3']], dim=1)
        
        x = self.deconv6(x)
        x = torch.cat([x, skip_connections['e2']], dim=1)
        
        x = self.deconv7(x)
        x = torch.cat([x, skip_connections['e1']], dim=1)
        
        return self.deconv8(x)