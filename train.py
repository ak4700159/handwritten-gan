import torch
from torch.utils.data import DataLoader
from main import *
from generator import FontGAN
from dataset import FontDataset
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.serialization import add_safe_globals
from embedding import *
from function import save_checkpoint




def train_font_gan(config: GANConfig, data_dir: str, save_dir: str, device: torch.device, 
                  start_epoch: int = 0, initial_model: Optional[FontGAN] = None):
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

    # 평가용 데이터로더
    val_dataset = FontDataset(data_dir, config.img_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # GAN 모델 초기화 또는 기존 모델 사용
    gan = initial_model if initial_model is not None else FontGAN(config, device)
    
    # 나머지 학습 코드는 동일하지만 epoch 범위 수정
    best_loss = float('inf')
    for epoch in range(start_epoch, start_epoch + config.max_epoch):
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
        
        # 에포크 평균 손실 계산
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}

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
        

        # 일정 주기로 체크포인트 저장
        if (epoch + 1) % config.model_save_step == 0:
            timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
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
        
        # 최고 성능 모델 저장
        current_loss = avg_losses['g_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")
            
    print("Training completed!")
    return gan