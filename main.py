import os
import torch
from pathlib import Path
from dataclasses import dataclass
import pickle
from embedding import generate_font_embeddings
from function import resume_training
from train import train_font_gan
from torch.serialization import add_safe_globals

@dataclass
class GANConfig:
    """Configuration for GAN training"""
    img_size: int = 128
    embedding_dim: int = 128
    conv_dim: int = 64
    batch_size: int = 16
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    max_epoch: int = 100
    schedule: int = 20
    l1_lambda: float = 100
    const_lambda: float = 15
    sample_step: int = 350
    model_save_step: int = 1
    fonts_num: int = 26  # 폰트 개수 추가
    log_step: int = 1  # 로깅 주기도 추가
    eval_step: int = 5  # 5 에포크마다 평가
    eval_samples: int = 100  # 평가할 샘플 수

# GANConfig를 안전한 전역 클래스로 등록
add_safe_globals([GANConfig])

if __name__ == "__main__":
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
    checkpoint_path = "./results/checkpoints/checkpoint_epoch_5_1124-2345.pth"  # 이전 체크포인트 경로
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        generate_font_embeddings(config.fonts_num, config.embedding_dim)

    # 체크포인트 경로가 있으면 학습 재개, 없으면 새로 시작
    if os.path.exists(checkpoint_path):
        gan = resume_training(checkpoint_path, config, data_dir, save_dir, device)
    else:
        print("\nNo checkpoint found. Starting new training...")
        gan = train_font_gan(config, data_dir, save_dir, device)