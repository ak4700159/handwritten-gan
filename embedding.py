import torch
import torch.nn.functional as F  # F로 자주 임포트되는 PyTorch의 함수형 인터페이스
from pathlib import Path


def generate_font_embeddings(
    fonts_num: int,
    embedding_dim: int = 128,
    save_dir: str = "./fixed_dir",
    stddev: float = 0.01
):
    """폰트 스타일 임베딩을 생성하고 저장하는 함수입니다.
    
    이 함수는 각 폰트에 대해 3x3 공간 구조를 가진 임베딩을 생성합니다.
    이는 한글 글자의 구조적 특성(초성/중성/종성)을 더 잘 표현할 수 있게 해줍니다.
    
    Args:
        fonts_num (int): 생성할 폰트 임베딩의 수
        embedding_dim (int): 각 임베딩의 차원 수
        save_dir (str): 임베딩을 저장할 디렉토리 경로
        stddev (float): 초기 랜덤 값의 표준편차
    
    Returns:
        torch.Tensor: 생성된 폰트 임베딩 텐서
    """
    # 먼저 1x1 크기로 임베딩을 생성합니다
    embeddings = torch.randn(fonts_num, 1, 1, embedding_dim) * stddev
    
    # L2 정규화를 수행하여 임베딩 벡터의 크기를 일정하게 만듭니다
    embeddings = F.normalize(embeddings.view(fonts_num, -1), p=2, dim=1)
    
    # 3x3 구조로 확장합니다
    embeddings = embeddings.view(fonts_num, embedding_dim, 1, 1)
    embeddings = F.interpolate(
        embeddings, 
        size=(3, 3), 
        mode='bilinear',
        align_corners=True
    )
    
    # 저장 경로를 설정하고 디렉토리를 생성합니다
    save_path = Path(save_dir) / 'EMBEDDINGS.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 임베딩을 안전하게 저장합니다
    torch.save(embeddings, save_path)
    
    print(f"Font style embeddings generated and saved to {save_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Each embedding represents a font style in a 3x3 spatial structure")
    
    return embeddings


def load_embeddings(embedding_path: str, device: torch.device):
    """안전하게 임베딩 로드"""
    try:
        # weights_only=True로 설정하여 안전하게 로드
        font_embeddings = torch.load(
            embedding_path,
            weights_only=True,
            map_location=device
        )
        
        # 텐서 타입 및 형태 확인
        if not isinstance(font_embeddings, torch.Tensor):
            raise TypeError("Loaded embeddings is not a torch.Tensor")
            
        print(f"Loaded font embeddings with shape: {font_embeddings.shape}")
        return font_embeddings
        
    except Exception as e:
        raise Exception(f"Error loading font embeddings: {e}")
