from dataclasses import dataclass
from torch.serialization import add_safe_globals


@dataclass
class GANConfig:
    """Configuration for GAN training"""
    img_size: int = 128
    embedding_dim: int = 128
    conv_dim: int = 128
    batch_size: int = 64
    lr: float = 0.0001
    beta1: float = 0.5
    beta2: float = 0.999
    max_epoch: int = 100
    schedule: int = 20
    l1_lambda: float = 80
    const_lambda: float = 10
    sample_step: int = 350
    model_save_step: int = 1
    fonts_num: int = 26  # 폰트 개수 추가
    log_step: int = 20  # 로깅 주기(몇 번째 배치마다)
    eval_step: int = 5 # 5 에포크마다 평가
    eval_samples: int = 10  # 평가할 샘플 수
    d_update_freq: int = 1   # 1번마다 한 번씩만 D 업데이트


# GANConfig를 안전한 전역 클래스로 등록
add_safe_globals([GANConfig])