import os
import torch
from pathlib import Path
import pickle
from embedding import generate_font_embeddings, load_embeddings
from torch.utils.data import DataLoader
from dataset import FontDataset
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from generator import FontGAN
from gan_config import GANConfig
import csv
from datetime import datetime  # datetime.datetime 대신 datetime만 import
from message import GANTrainingCallback, DiscordLogger, setup_training_callback
from torchvision.utils import save_image
import numpy as np
import os
from PIL import Image

def resume_training(checkpoint_path: str, config: GANConfig, data_dir: str, save_dir: str, device: torch.device):
    """이전 체크포인트에서 학습 재개"""
    print(f"\n=== Resuming training from {checkpoint_path} ===")
    
    # GAN 모델 초기화
    gan = FontGAN(config, device)
    
    # 체크포인트 로드
    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
        
        # 모델 가중치 로드
        gan.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        gan.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # 옵티마이저 상태 로드
        gan.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        gan.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        # 이전 학습 정보
        start_epoch = checkpoint['epoch'] + 1
        prev_losses = checkpoint.get('losses', {})
        print(f"Previous training stopped at epoch {start_epoch}")
        if prev_losses:
            print("Previous losses:", prev_losses)
        
        # 남은 에포크 수 계산
        remaining_epochs = config.max_epoch - start_epoch
        if remaining_epochs <= 0:
            print("Warning: No remaining epochs. Consider increasing max_epoch in config.")
            return gan
            
        # config 업데이트
        new_config = GANConfig(
            **{**config.__dict__,
            'max_epoch': remaining_epochs}
        )
        
        # 학습 재개
        print(f"\nResuming training for {remaining_epochs} more epochs...")
        return train_font_gan(new_config, data_dir, save_dir, device, 
                            start_epoch=start_epoch,
                            initial_model=gan)
                            
    except Exception as e:
        print(f"Failed to resume training: {e}")
        print("Starting new training session...")
        return train_font_gan(config, data_dir, save_dir, device)
    
def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """체크포인트를 안전하게 로드"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 필요한 정보만 추출
        safe_checkpoint = {
            'encoder_state_dict': checkpoint['encoder_state_dict'],
            'decoder_state_dict': checkpoint['decoder_state_dict'],
            'discriminator_state_dict': checkpoint['discriminator_state_dict'],
            'g_optimizer_state_dict': checkpoint['g_optimizer_state_dict'],
            'd_optimizer_state_dict': checkpoint['d_optimizer_state_dict'],
            'epoch': checkpoint['epoch'],
            'losses': checkpoint.get('losses', {})
        }
        
        return safe_checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def save_checkpoint(model: FontGAN, epoch: int, losses: dict, save_path: Path):
    """체크포인트를 안전하게 저장"""
    # 모델 상태만 저장
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'g_optimizer_state_dict': model.g_optimizer.state_dict(),
        'd_optimizer_state_dict': model.d_optimizer.state_dict(),
        'losses': losses
    }
    
    try:
        # weights_only 매개변수 없이 저장
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

def train_font_gan(config: GANConfig, data_dir: str, save_dir: str, device: torch.device, 
                  start_epoch: int = 0, initial_model: Optional[FontGAN] = None, 
                  callback: GANTrainingCallback = None,
                  pretrained_path: str = None):  # 사전 학습된 모델 경로 추가
    """폰트 GAN 학습 함수 - 전이학습 지원 추가"""
    print("\n=== Starting Font GAN Transfer Learning ===")
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Starting from epoch: {start_epoch}")
    
    # 저장 디렉토리 설정
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / 'checkpoints'
    sample_dir = save_dir / 'samples'
    
    for dir_path in [checkpoint_dir, sample_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 폰트 임베딩 로드
    embedding_path = Path("./fixed_dir/EMBEDDINGS.pkl")
    if not embedding_path.exists():
        raise FileNotFoundError(f"Font embeddings not found at {embedding_path}")
    
    font_embeddings = load_embeddings(embedding_path, device)
    
    # GAN 모델 초기화 또는 로드
    if initial_model is None:
        gan = FontGAN(config, device)
        if pretrained_path:  # 사전 학습된 모델이 제공된 경우
            print(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            
            # 모델 가중치 로드
            gan.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            gan.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            
            # 인코더 고정 (전이학습을 위해)
            for param in gan.encoder.parameters():
                param.requires_grad = False
            gan.encoder.eval()
            
            print("Encoder frozen for transfer learning")
            
            # 옵티마이저 재설정 (디코더만을 위한)
            gan.g_optimizer = torch.optim.Adam(
                gan.decoder.parameters(),
                lr=config.lr,
                betas=(config.beta1, config.beta2)
            )
    else:
        gan = initial_model
    
    # 데이터 로더 설정
    dataset = FontDataset(data_dir, config.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 학습 상태 추적을 위한 변수들
    best_loss = float('inf')
    patience_counter = 0
    
    # 학습 루프
    for epoch in range(start_epoch, start_epoch + config.max_epoch):
        print(f"\n=== Epoch {epoch+1}/{config.max_epoch + start_epoch} ===")
        epoch_losses = {
            'g_loss': [], 'd_loss': [], 
            'l1_loss': [], 'const_loss': [],
            'cat_loss': []
        }
        
        # 배치 학습
        for batch_idx, (source, target, font_ids) in enumerate(dataloader):
            # 학습 단계
            losses = gan.train_step(source, target, font_embeddings, font_ids)
            
            # 손실값 기록
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            # 진행상황 출력
            if batch_idx % config.log_step == 0:
                log_str = f"Epoch [{epoch+1}/{config.max_epoch + start_epoch}], "
                log_str += f"Batch [{batch_idx+1}/{len(dataloader)}], "
                log_str += ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                print(log_str)
        
        # 에포크 평균 손실 계산
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        
        # 샘플 이미지 생성 및 저장
        if (epoch + 1) % 5 == 0:
            gan.save_samples(
                sample_dir / f'transfer_samples_epoch_{epoch+1}.png',
                source[:4].to(device), 
                target[:4].to(device), 
                font_ids[:4].to(device),
                font_embeddings
            )
        
        # 체크포인트 저장
        if avg_losses['g_loss'] < best_loss:
            best_loss = avg_losses['g_loss']
            save_checkpoint(gan, epoch, avg_losses, 
                          checkpoint_dir / 'best_transfer_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:  # 조기 종료 조건
                print("\nEarly stopping triggered")
                break
    
    print("Transfer learning completed!")
    return gan
def main():
    # 기존의 설정을 최대한 활용하되, 전이학습에 맞게 일부 수정
    config = GANConfig(
        img_size=128,
        embedding_dim=128,
        batch_size=32,        # 기존 배치 크기 유지
        max_epoch=100,        # 기존 에포크 수 유지
        fonts_num=26,         # 기존 폰트 수 유지
        lr=0.0001,           # 기존 학습률 유지
        schedule=20,          # 기존 스케줄링 간격 유지
        l1_lambda=80,        # 스타일 보존을 위한 L1 가중치
        const_lambda=10,     # 일관성 유지를 위한 가중치
        eval_step=5          # 평가 주기 유지
    )
    
    # 경로 설정
    data_dir = "./dataset"
    save_dir = "./final_data"
    checkpoint_path = "./results/checkpoints/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 먼저 체크포인트를 로드하여 손실값 확인
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # 전이학습을 위한 모델 초기화
    gan = FontGAN(config, device)
    
    # 사전 학습된 가중치 로드
    gan.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    gan.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # 인코더 고정
    for param in gan.encoder.parameters():
        param.requires_grad = False
    gan.encoder.eval()
    
    # 디코더만을 위한 옵티마이저 재설정 (g_optimizer 재정의)
    gan.g_optimizer = torch.optim.Adam(
        gan.decoder.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2)
    )
    
    # 기존의 train_font_gan 함수 호출
    gan = train_font_gan(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        device=device,
        start_epoch=0,
        initial_model=gan  # 이미 설정된 모델 전달
    )
    
    return gan

if __name__ == "__main__":
    main()
# ==========================================================================================================================
# ==========================================================================================================================
# =========================================================================================================================