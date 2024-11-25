import torch
import torch.nn.functional as F  # F로 자주 임포트되는 PyTorch의 함수형 인터페이스
from pathlib import Path


def generate_font_embeddings(
    fonts_num: int,
    embedding_dim: int = 128,
    save_dir: str = "./fixed_dir",
    stddev: float = 0.01
):
    """Generate and save font style embeddings"""
    # fonts_num + 1 크기로 생성 (0번 인덱스는 패딩용)
    embeddings = torch.randn(fonts_num, 1, 1, embedding_dim) * stddev
    
    # L2 normalize embeddings
    embeddings = F.normalize(embeddings.view(fonts_num, -1), p=2, dim=1)
    embeddings = embeddings.view(fonts_num, 1, 1, embedding_dim)
    
    # 저장
    save_path = Path(save_dir) / 'EMBEDDINGS.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 안전하게 저장
    torch.save(embeddings, save_path, weights_only=True)
    
    print(f"Font style embeddings generated and saved to {save_path}")
    print(f"Shape: {embeddings.shape}")
    
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

#     # stddev = 표준 편차
#     # 임베딩 파일이 없으면 초기값 넣어 생성한다?
# def init_embedding(embedding_num, embedding_dim, stddev=0.01):
#     # 정규분포를 생성하는 난수 함수, 정규분포에서 무작위 난수를 생성하여 텐서를 반환
#     # embedding_num * embedding_dim 차원의 텐서가 생성
#     embedding = torch.randn(embedding_num, embedding_dim) * stddev
    
#     embedding = embedding.reshape((embedding_num, 1, 1, embedding_dim))
#     return embedding

# def generate_font_embeddings(fonts_num, embedding_dim=128, save_dir="./", stddev=0.01):
#     """
#     fonts_num: 학습할 폰트 스타일의 수
#     embedding_dim: 임베딩 벡터의 차원
#     save_dir: 임베딩 파일을 저장할 경로
#     stddev: 정규분포의 표준편차
#     """
#     # 초기 임베딩 생성 (fonts_num x 1 x 1 x embedding_dim)
#     embeddings = init_embedding(fonts_num, embedding_dim, stddev=stddev)
    
#     # L2 정규화 적용 - 각 폰트 스타일의 임베딩 벡터를 단위 벡터로 정규화
#     embeddings_flat = embeddings.view(fonts_num, -1)
#     embeddings_normalized = F.normalize(embeddings_flat, p=2, dim=1)
#     embeddings = embeddings_normalized.view(fonts_num, 1, 1, embedding_dim)
    
#     # 저장
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     save_path = os.path.join(save_dir, 'EMBEDDINGS.pkl')
#     torch.save(embeddings, save_path)
    
#     print(f"Font style embeddings generated and saved to {save_path}")
#     print(f"Shape: {embeddings.shape}")
    
#     return embeddings