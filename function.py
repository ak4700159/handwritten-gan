import torch
from pathlib import Path
from generator import FontGAN
from embedding import *
from main import GANConfig

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