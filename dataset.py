from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import torch


class FontDataset:
    """Enhanced dataset class for font images"""
    def __init__(
        self,
        data_dir: str,
        img_size: int = 128,
        resize_fix: int = 90,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.resize_fix = resize_fix
        self.augment = augment
        
        # Load and process data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Tuple[int, np.ndarray]]:
        """Load and preprocess image data"""
        processed_data = []
        try:
            with open(self.data_dir / "handwritten_train.pkl", "rb") as f:
                while True:
                    try:
                        # 데이터를 하나씩 로드
                        example = pickle.load(f)
                        if not example:
                            continue
                            
                        # 데이터 구조 확인 및 처리
                        if len(example) >= 2:  # 최소 (font_id, image_data) 형식
                            font_id = example[0]
                            img_data = example[-1]  # 마지막 요소가 이미지 데이터
                            
                            # 이미지 처리
                            source_img, target_img = self._process_image_pair(img_data)
                            processed_data.append((font_id, source_img, target_img))
                            
                    except EOFError:
                        break
                    except Exception as e:
                        print(f"Error processing data: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            
        if not processed_data:
            raise ValueError("No data could be loaded from the pickle file")
            
        print(f"Loaded {len(processed_data)} samples")
        return processed_data

    def _process_image_pair(self, img_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Process a pair of source and target images"""
        try:
            # Convert bytes to PIL Image
            img = Image.open(BytesIO(img_data))
            img_array = np.array(img)
            
            # Split into source and target
            w = img_array.shape[1]
            source_img = img_array[:, w//2:]  # 오른쪽 절반이 소스
            target_img = img_array[:, :w//2]  # 왼쪽 절반이 타겟
            
            # Center and resize images
            source_img = self._center_and_resize(source_img)
            target_img = self._center_and_resize(target_img)
            
            return source_img, target_img
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # 오류 발생시 기본 이미지 반환
            blank = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            return blank, blank

    def _center_and_resize(self, img: np.ndarray) -> np.ndarray:
        """Center and resize image with error handling"""
        try:
            # Convert to PIL Image for processing
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
                
            # Get dimensions
            w, h = img.size
            
            # Calculate new dimensions
            if h > w:
                new_h = self.resize_fix
                new_w = int(w * (new_h / h))
            else:
                new_w = self.resize_fix
                new_h = int(h * (new_w / w))
                
            # Resize
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Create blank canvas
            new_img = Image.new('L', (self.img_size, self.img_size), 255)
            
            # Calculate paste position
            paste_x = (self.img_size - new_w) // 2
            paste_y = (self.img_size - new_h) // 2
            
            # Paste resized image
            new_img.paste(resized, (paste_x, paste_y))
            
            # Convert to numpy and normalize
            result = np.array(new_img, dtype=np.float32)
            result = (result / 127.5) - 1.0
            
            return result
            
        except Exception as e:
            print(f"Error in center_and_resize: {e}")
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        font_id, source_img, target_img = self.data[idx]
        
        # Convert to tensor
        source_tensor = torch.from_numpy(source_img).unsqueeze(0)
        target_tensor = torch.from_numpy(target_img).unsqueeze(0)
        
        return source_tensor, target_tensor, font_id