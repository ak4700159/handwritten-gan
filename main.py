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
import pandas as pd
import csv
from datetime import datetime  # datetime.datetime 대신 datetime만 import
from message import GANTrainingCallback, DiscordLogger

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
                  start_epoch: int = 0, initial_model: Optional[FontGAN] = None, callback: GANTrainingCallback = None):
    print("\n=== Starting Font GAN Training ===")
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Starting from epoch: {start_epoch}")
    
    # 저장 디렉토리 생성
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / 'checkpoints'
    sample_dir = save_dir / 'samples'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 폰트 임베딩 로드
    embedding_path = Path("./fixed_dir/EMBEDDINGS.pkl")
    if not embedding_path.exists():
        raise FileNotFoundError(f"Font embeddings not found at {embedding_path}")
    
    font_embeddings = load_embeddings(embedding_path, device)
    
    # 학습용 데이터 로더 설정
    dataset = FontDataset(data_dir, config.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 평가용 데이터로더 설정
    val_dataset = FontDataset(data_dir, config.img_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # GAN 모델 초기화 또는 기존 모델 사용
    gan = initial_model if initial_model is not None else FontGAN(config, device)
    
    # 조기 종료를 위한 파라미터 설정
    early_stopping_patience = 15  # 15 에폭 동안 개선이 없으면 종료
    min_loss_improvement = 0.001  # 최소 손실 개선 기준값
    best_loss = float('inf')
    patience_counter = 0

    # CSV 파일 경로 설정
    timestamp = datetime.now().strftime("%m%d-%H%M")
    metrics_dir = Path(save_dir) / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    loss_file = metrics_dir / f'training_losses_{timestamp}.csv'
    eval_file = metrics_dir / f'evaluation_metrics_{timestamp}.csv'
    
    # CSV 헤더 작성
    with open(loss_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'd_loss', 'g_loss', 'l1_loss', 'const_loss'])
        
    with open(eval_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'l1_loss', 'const_loss', 'discriminator_acc', 'font_classification_acc'])

        # 학습률 스케줄러 설정
    schedulers = {
        'generator': torch.optim.lr_scheduler.StepLR(
            gan.g_optimizer,
            step_size=config.schedule,
            gamma=0.5  # 학습률을 절반으로 감소
        ),
        'discriminator': torch.optim.lr_scheduler.StepLR(
            gan.d_optimizer,
            step_size=config.schedule,
            gamma=0.5
        )
    }

    if callback:
        callback.on_training_start(config)

    for epoch in range(start_epoch, start_epoch + config.max_epoch):
        if callback:
            callback.on_epoch_start(epoch+1, config.max_epoch)
            callback.reset_batch_history()  # 새 에폭 시작시 배치 히스토리 초기화

        print(f"\n=== Epoch {epoch+1}/{config.max_epoch + start_epoch} ===")
        epoch_losses = {
            'g_loss': [], 
            'd_loss': [], 
            'l1_loss': [],
            'const_loss': []
        }
        
        for batch_idx, (source, target, font_ids) in enumerate(dataloader):
            losses = gan.train_step(source, target, font_embeddings, font_ids)
                
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            if batch_idx % config.log_step == 0:
                log_str = f"Epoch [{epoch+1}/{config.max_epoch}], "
                log_str += f"Batch [{batch_idx+1}/{len(dataloader)}], "
                log_str += ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                print(log_str)
                
                # CSV에 손실값 기록
                with open(loss_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        batch_idx + 1,
                        losses['d_loss'],
                        losses['g_loss'],
                        losses['l1_loss'],
                        losses['const_loss']
                    ])
            
            if callback:
                callback.on_batch_end(epoch+1, batch_idx, len(dataloader), losses)
        
        # 에포크 평균 손실 계산
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}

        # 학습률 조정
        schedulers['generator'].step()
        schedulers['discriminator'].step()

        if callback:
            callback.on_epoch_end(epoch+1, avg_losses, sample_dir)
        
        def create_checkpoint(gan, epoch, avg_losses, font_embeddings):
            return {
                'epoch': epoch,
                'encoder_state_dict': gan.encoder.state_dict(),
                'decoder_state_dict': gan.decoder.state_dict(),
                'discriminator_state_dict': gan.discriminator.state_dict(),
                'g_optimizer_state_dict': gan.g_optimizer.state_dict(),
                'd_optimizer_state_dict': gan.d_optimizer.state_dict(),
                'config': config,
                'losses': avg_losses,
                'font_embeddings': font_embeddings
            }        
        
        # 현재 체크포인트 생성
        checkpoint = create_checkpoint(gan, epoch, avg_losses, font_embeddings)

        # 주기적으로 평가 수행
        if (epoch + 1) % config.eval_step == 0:
            print(f"\nEvaluating model at epoch {epoch + 1}...")
            metrics = gan.evaluate_metrics(val_loader, font_embeddings)
            
            # 평가 샘플 생성
            eval_dir = Path(save_dir) / 'evaluation' / f'epoch_{epoch+1}'
            gan.generate_evaluation_samples(val_loader, font_embeddings, eval_dir)

            # CSV에 평가 지표 기록
            with open(eval_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    metrics['l1_loss'],
                    metrics['const_loss'],
                    metrics['discriminator_acc'],
                    metrics['font_classification_acc']
                ])
        

        # 일정 주기로 체크포인트 저장
        if (epoch + 1) % config.model_save_step == 0:
            timestamp = datetime.now().strftime("%m%d-%H%M")
            save_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}_{timestamp}.pth'
            save_checkpoint(gan, epoch, avg_losses, save_path)
            
            # 샘플 이미지 생성 및 저장
            gan.save_samples(
                sample_dir / f'samples_epoch_{epoch+1}_{timestamp}.png',
                source[:8].to(device), 
                target[:8].to(device), 
                font_ids[:8].to(device),
                font_embeddings
            )
            if callback:
                callback.on_epoch_end(epoch+1, avg_losses, sample_path=sample_dir)
        
        # 최고 성능 모델 저장
        current_loss = avg_losses['g_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")

        # 조기 종료 검사
        current_loss = avg_losses['g_loss']
        if current_loss < best_loss - min_loss_improvement:
            best_loss = current_loss
            patience_counter = 0
            # 최고 성능 모델 저장
            best_model_path = Path(save_dir) / 'checkpoints' / 'best_model.pth'
            save_checkpoint(gan, epoch, avg_losses, best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best loss achieved: {best_loss:.4f}")
                break
        
    # 학습이 종료됨을 알림
    if callback:
        callback.on_training_end()

    print("Training completed!")
    return gan

if __name__ == "__main__":
            # 디스코드 봇 연결
    TOKEN = ''
    TARGET_USER_ID = 0  # 대상 유저의 ID를 입력하세요

    # Configuration
    config = GANConfig(
        img_size=128,
        embedding_dim=128,
        batch_size=32,
        max_epoch=100,
        fonts_num=26
    )
    
    # Paths
    data_dir = "./dataset"
    save_dir = "./results"
    checkpoint_path = "./results/checkpoints/checkpoint_epoch_60_1128-1815.pth"  # 이전 체크포인트 경로
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    discord_logger = DiscordLogger(TOKEN, TARGET_USER_ID)
    callback = GANTrainingCallback(discord_logger, save_dir)

    # 실제 폰트 수 확인
    try:
        with open(os.path.join(data_dir, "train.pkl"), "rb") as f:
            data = pickle.load(f)
            if isinstance(data, list) and len(data) > 0:
                font_ids = set([item[0] for item in data])
                actual_fonts_num = len(font_ids)
                print(f"Detected {actual_fonts_num} unique fonts")
                config.fonts_num = actual_fonts_num
    except Exception as e:
        print(f"Warning: Could not automatically detect number of fonts: {e}")
        print("Using default value:", config.fonts_num)

    # Generate embeddings if needed
    if not (Path("./fixed_dir") / "EMBEDDINGS.pkl").exists():
        print("새로운 임베딩 생성")
        generate_font_embeddings(config.fonts_num, config.embedding_dim)

    # 체크포인트 경로가 있으면 학습 재개, 없으면 새로 시작
    if os.path.exists(checkpoint_path):
        gan = resume_training(checkpoint_path, config, data_dir, save_dir, device)
    else:
        print("\nNo checkpoint found. Starting new training...")
        gan = train_font_gan(config, data_dir, save_dir, device, callback=callback)