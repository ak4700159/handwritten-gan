from gan_config import GANConfig
from layer import *
from discriminator import Discriminator
from torchvision.utils import save_image
import torch.nn.functional as F
from pathlib import Path



class FontGAN:
    def __init__(self, config: GANConfig, device: torch.device):
        self.config = config
        self.device = device
        self.train_step_count = 0
        
        # Initialize networks
        self.encoder = Encoder(conv_dim=config.conv_dim).to(device)
        # embedded_dim을 encoded_source의 채널 수 + embedding_dim으로 설정
        self.decoder = Decoder(
            embedded_dim=1152,  # 인코더 출력(512) + 임베딩(512)
            conv_dim=config.conv_dim
        ).to(device)
        self.discriminator = Discriminator(category_num=config.fonts_num).to(device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.lr * 0.2,
            betas=(config.beta1, config.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr * 0.5,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss().to(device)
        self.bce_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

    def eval(self):
        """평가 모드로 전환"""
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

    def train(self, mode=True):
        """학습 모드로 전환"""
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.discriminator.train(mode)

    def save_samples(self, save_path: str, source: torch.Tensor, 
                    target: torch.Tensor, font_ids: torch.Tensor,
                    font_embeddings: torch.Tensor):
        """Generate and save sample images"""
        # 이전 모드 저장
        prev_encoder_mode = self.encoder.training
        prev_decoder_mode = self.decoder.training
        prev_discriminator_mode = self.discriminator.training
        
        # 평가 모드로 전환
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        
        try:
            with torch.no_grad():
                # 가짜 이미지 생성
                encoded_source, skip_connections = self.encoder(source)
                embedding = self._get_embeddings(font_embeddings, font_ids)
                embedded = torch.cat([encoded_source, embedding], dim=1)
                fake_target = self.decoder(embedded, skip_connections)
                
                # 이미지 그리드 생성
                source = (source + 1) / 2  # [-1, 1] -> [0, 1] 범위로 변환
                target = (target + 1) / 2
                fake_target = (fake_target + 1) / 2
                
                # 샘플 이미지 저장
                comparison = torch.cat([source, target, fake_target], dim=3)
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                save_image(comparison, save_path, nrow=4, normalize=False)
                print(f"Samples saved to {save_path}")
                
        except Exception as e:
            print(f"Error in save_samples: {e}")
            raise
            
        finally:
            # 이전 모드로 복원
            self.encoder.train(prev_encoder_mode)
            self.decoder.train(prev_decoder_mode)
            self.discriminator.train(prev_discriminator_mode)
        self.train()  # 학습 모드로 복귀
        
    def train_step(self, real_source, real_target, font_embeddings, font_ids):
        # batch_size = real_source.size(0)
        # print(f"\n===== New Training Step =====")
        # print(f"Batch size: {batch_size}")
        
        # 모든 입력 데이터를 GPU로 이동
        real_source = real_source.to(self.device)
        real_target = real_target.to(self.device)
        font_embeddings = font_embeddings.to(self.device)
        font_ids = font_ids.to(self.device)  # font_ids도 GPU로 이동


        # 인코더는 고정된 상태로 특징 추출
        with torch.no_grad():
            encoded_source, skip_connections = self.encoder(real_source)
        # print(f"Device check - Source: {real_source.device}, Target: {real_target.device}")
        # print(f"Device check - Embeddings: {font_embeddings.device}, Font IDs: {font_ids.device}")
        
        # Move data to device
        real_source = real_source.to(self.device)
        real_target = real_target.to(self.device)
        font_embeddings = font_embeddings.to(self.device)
        
        # print(f"Input shapes - Source: {real_source.shape}, Target: {real_target.shape}")
        
        # Generate fake image
        # print("\n1. Encoder Feature Extraction:")
        encoded_source, skip_connections = self.encoder(real_source)
        # print(f"Encoded source shape: {encoded_source.shape}")
        # print("Skip connection shapes:")
        # for k, v in skip_connections.items():
        #     print(f"  {k}: {v.shape}")
        
        # Get font embedding
        # print("\n2. Font Embedding Processing:")
        embedding = self._get_embeddings(font_embeddings, font_ids)
        # print(f"Font embedding shape: {embedding.shape}")
        
        # Concatenate features and embedding
        # print("\n3. Feature Concatenation:")
        embedded = torch.cat([encoded_source, embedding], dim=1)
        # print(f"Combined embedded shape: {embedded.shape}")
        
        # Generate fake image
        # print("\n4. Decoder Generation:")
        fake_target = self.decoder(embedded, skip_connections)
        # print(f"Generated fake target shape: {fake_target.shape}")
        
        # Discriminator 업데이트 (빈도 조절)
        if self.train_step_count % self.config.d_update_freq == 0:
            d_real_score, d_real_logits, d_real_cat = self.discriminator(
                torch.cat([real_source, real_target], dim=1)
            )
            d_fake_score, d_fake_logits, d_fake_cat = self.discriminator(
                torch.cat([real_source, fake_target.detach()], dim=1)
            )
            
            # Label smoothing 적용
            real_labels = torch.ones_like(d_real_logits).to(self.device) * 0.9  # 1.0 대신 0.9
            fake_labels = torch.zeros_like(d_fake_logits).to(self.device) * 0.1  # 0.0 대신 0.1
            
            d_loss = self._adversarial_loss(d_real_logits, d_fake_logits, real_labels, fake_labels)
            d_cat_loss = self._category_loss(d_real_cat, d_fake_cat, font_ids)
            
            d_total_loss = d_loss + d_cat_loss
            
            self.d_optimizer.zero_grad()
            d_total_loss.backward()
            self.d_optimizer.step()
        
        
        # Generator 업데이트
        g_fake_score, g_fake_logits, g_fake_cat = self.discriminator(
            torch.cat([real_source, fake_target], dim=1)
        )
        
        # Generator loss 계산에 Feature matching 추가
        g_adv_loss = self.bce_loss(g_fake_logits, torch.ones_like(g_fake_logits))
        g_l1_loss = self.l1_loss(fake_target, real_target) * self.config.l1_lambda
        g_const_loss = self._consistency_loss(encoded_source, fake_target)
        g_cat_loss = self.bce_loss(g_fake_cat, F.one_hot(font_ids, self.config.fonts_num).float())
        
        g_total_loss = g_adv_loss + g_l1_loss + g_const_loss + g_cat_loss
        
        # Update generator
        self.g_optimizer.zero_grad()
        g_total_loss.backward()
        self.g_optimizer.step()
        # print(f"Generator Losses - Adversarial: {g_adv_loss.item():.4f}, L1: {g_l1_loss.item():.4f}")
        # print(f"                 Consistency: {g_const_loss.item():.4f}, Category: {g_cat_loss.item():.4f}")

        # print("Generator update completed")
        
        # print("\n===== Training Step Completed =====")
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_total_loss.item(),
            'l1_loss': g_l1_loss.item(),
            'const_loss': g_const_loss.item(),
            'cat_loss' : g_cat_loss.item(),
        }
    
    
    def _get_embeddings(self, embeddings: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Get font embeddings for given IDs"""
        try:
            # font_ids를 0-based 인덱스로 변환
            adjusted_ids = ids - 1
            
            # 유효한 인덱스 범위 체크
            if torch.any(adjusted_ids < 0) or torch.any(adjusted_ids >= embeddings.size(0)):
                raise ValueError(f"Font IDs out of range. Expected 1 to {embeddings.size(0)}, got min={ids.min()}, max={ids.max()}")
            
            # 임베딩을 선택 (이미 3x3 구조)
            selected = embeddings[adjusted_ids]  # Shape: [batch_size, embedding_dim, 3, 3]
            
            # 디버깅을 위한 shape 출력
            # print(f"Selected embedding shape: {selected.shape}")
            
            return selected
            
        except Exception as e:
            print(f"Error in _get_embeddings: {e}")
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Font IDs: {ids}")
            raise
            
    def _adversarial_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor, 
                        real_labels: torch.Tensor, fake_labels: torch.Tensor) -> torch.Tensor:
        """Calculate adversarial loss for discriminator with label smoothing"""
        real_loss = self.bce_loss(real_logits, real_labels)
        fake_loss = self.bce_loss(fake_logits, fake_labels)
        
        return (real_loss + fake_loss) * 0.5
    
    def _category_loss(self, real_cat: torch.Tensor, fake_cat: torch.Tensor, font_ids: torch.Tensor) -> torch.Tensor:
        """Calculate category classification loss"""
        # font_ids를 GPU로 이동
        font_ids = font_ids.to(self.device)
        
        # one-hot 인코딩을 GPU에서 직접 생성
        real_labels = F.one_hot(font_ids, self.config.fonts_num).float().to(self.device)
        
        # print(f"Device check - real_cat: {real_cat.device}, real_labels: {real_labels.device}")
        
        real_loss = self.bce_loss(real_cat, real_labels)
        fake_loss = self.bce_loss(fake_cat, real_labels)
        
        return (real_loss + fake_loss) * 0.5
    
    def _consistency_loss(self, encoded_source: torch.Tensor, fake_target: torch.Tensor) -> torch.Tensor:
        """Calculate consistency loss between encoded source and encoded fake"""
        # Encode the generated image
        encoded_fake, _ = self.encoder(fake_target)
        return self.mse_loss(encoded_source, encoded_fake) * self.config.const_lambda

    def verify_frozen_encoder(self):
        """인코더가 제대로 고정되었는지 확인하는 함수"""
        frozen = all(not p.requires_grad for p in self.encoder.parameters())
        trainable = any(p.requires_grad for p in self.decoder.parameters())
        
        print("=== Transfer Learning Setup Verification ===")
        print(f"Encoder frozen: {frozen}")
        print(f"Decoder trainable: {trainable}")
        print(f"Number of trainable parameters in decoder: {sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}")
        print("=========================================")

    def setup_transfer_learning(self, pretrained_path: str):
        """전이학습을 위한 모델 설정"""
        # 사전 학습된 모델 로드
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        # 모델 가중치 로드
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # 인코더 파라미터 고정
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # 옵티마이저 재설정 (디코더와 판별자만 학습)
        self.g_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.config.lr * 0.1,  # 더 작은 학습률 사용
            betas=(self.config.beta1, self.config.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr * 0.1,
            betas=(self.config.beta1, self.config.beta2)
        )
        self.verify_frozen_encoder()

    def evaluate_metrics(self, dataloader, font_embeddings):
        """모델 성능 평가"""
        self.eval()
        metrics = {
            'l1_loss': [],
            'const_loss': [],
            'discriminator_acc': [],
            'font_classification_acc': []
            # FID score는 복잡한 계산이 필요하므로 일단 제외
        }
        
        try:
            with torch.no_grad():
                # 데이터 로더가 비어있는지 확인
                if len(dataloader) == 0:
                    raise ValueError("Dataloader is empty")
                    
                for batch_idx, (source, target, font_ids) in enumerate(dataloader):
                    if batch_idx >= 100:  # 평가할 배치 수 제한
                        break
                        
                    source = source.to(self.device)
                    target = target.to(self.device)
                    font_ids = font_ids.to(self.device)
                    
                    # 가짜 이미지 생성
                    encoded_source, skip_connections = self.encoder(source)
                    embedding = self._get_embeddings(font_embeddings, font_ids)
                    embedded = torch.cat([encoded_source, embedding], dim=1)
                    fake_target = self.decoder(embedded, skip_connections)
                    
                    # L1 Loss 계산
                    l1_loss = self.l1_loss(fake_target, target)
                    metrics['l1_loss'].append(l1_loss.item())
                    
                    # Consistency Loss 계산
                    const_loss = self._consistency_loss(encoded_source, fake_target)
                    metrics['const_loss'].append(const_loss.item())
                    
                    # Discriminator 정확도 계산
                    real_score, _, real_cat = self.discriminator(torch.cat([source, target], dim=1))
                    fake_score, _, fake_cat = self.discriminator(torch.cat([source, fake_target], dim=1))
                    
                    disc_acc = ((real_score > 0.5).float().mean() + 
                            (fake_score < 0.5).float().mean()) / 2
                    metrics['discriminator_acc'].append(disc_acc.item())
                    
                    # 폰트 분류 정확도 계산
                    font_labels = F.one_hot(font_ids, self.config.fonts_num).float()
                    font_acc = (torch.argmax(real_cat, dim=1) == font_ids).float().mean()
                    metrics['font_classification_acc'].append(font_acc.item())
                    
                # 각 메트릭이 비어있지 않은지 확인
                for k, v in metrics.items():
                    if not v:
                        print(f"Warning: No values collected for metric {k}")
                        metrics[k] = [0.0]  # 기본값 설정
                
                # 평균 계산
                avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
                
                print("\nEvaluation Metrics:")
                print(f"L1 Loss (픽셀 유사도): {avg_metrics['l1_loss']:.4f}")
                print(f"Consistency Loss (특징 보존): {avg_metrics['const_loss']:.4f}")
                print(f"Discriminator Accuracy: {avg_metrics['discriminator_acc']:.4f}")
                print(f"Font Classification Accuracy: {avg_metrics['font_classification_acc']:.4f}")
                
                return avg_metrics
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {k: 0.0 for k in metrics.keys()}  # 오류 시 기본값 반환
            
        finally:
            self.train()

    def generate_evaluation_samples(self, dataloader, font_embeddings, save_dir: Path):
        self.eval()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 데이터셋에서 서로 다른 폰트를 가진 샘플 수집
        unique_samples = {}
        
        with torch.no_grad():
            for source, target, font_ids in dataloader:
                for i, font_id in enumerate(font_ids):
                    font_id = font_id.item()
                    if font_id not in unique_samples and len(unique_samples) < 10:
                        unique_samples[font_id] = (source[i:i+1], target[i:i+1])
                    if len(unique_samples) == 10:
                        break
                if len(unique_samples) == 10:
                    break
            
            # 수집된 샘플로 이미지 생성
            for idx, (font_id, (source, target)) in enumerate(unique_samples.items()):
                source = source.to(self.device)
                target = target.to(self.device)
                font_id = torch.tensor([font_id], device=self.device)
                
                # encoded_source, skip_connections = self.encoder(source)
                # embedding = self._get_embeddings(font_embeddings, font_id)
                # embedded = torch.cat([encoded_source, embedding], dim=1)
                # fake_target = self.decoder(embedded, skip_connections)
                
                self.save_samples(
                    save_dir / f'eval_sample_font_{font_id.item()}.png',
                    source,
                    target,
                    font_id,
                    font_embeddings
                )