import os
from pathlib import Path
import numpy as np
from PIL import Image  # pillow
from tqdm import tqdm 


class HandwritingPreprocessor:
    def __init__(self, input_dir: str, output_dir: str, font_id: int = 26):
        """손글씨 이미지 전처리기
        
        Args:
            input_dir: 원본 이미지가 있는 디렉토리
            output_dir: 처리된 이미지를 저장할 디렉토리
            font_id: 지정할 폰트 ID (기본값: 26)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.font_id = font_id
        self.img_size = 128
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_images(self):
        """모든 이미지 처리"""
        image_files = list(self.input_dir.glob("*.png"))
        
        print(f"Found {len(image_files)} images to process")
        successfully_processed = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # 원본 이미지 로드
                original = Image.open(img_path)
                
                # 유니코드 값 추출 (예: '0_12345.png' -> '12345')
                unicode_value = self._get_unicode_from_filename(img_path.name)
                
                # 새 파일명 생성 (26_12345.png)
                new_filename = f"{self.font_id}_{unicode_value}.png"
                
                # 이미지 처리 및 저장
                combined = self._create_combined_image(
                    self._extract_gothic(original),
                    self._extract_handwriting(original)
                )
                
                # 저장
                combined.save(self.output_dir / new_filename)
                successfully_processed += 1
                
            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successfully_processed} / {len(image_files)} images")
    
    def _extract_handwriting(self, image: Image.Image) -> Image.Image:
        """이미지의 오른쪽 절반(손글씨 부분) 추출"""
        width, height = image.size
        right_half = image.crop((width//2, 0, width, height))
        return right_half
    
    def _extract_gothic(self, image: Image.Image) -> Image.Image:
        """이미지의 왼쪽 절반(고딕체 부분) 추출"""
        width, height = image.size
        left_half = image.crop((0, 0, width//2, height))
        return left_half
    
    def _get_unicode_from_filename(self, filename: str) -> str:
        """파일명에서 유니코드 값 추출
        입력 형식: '0_12345.png'
        출력 형식: '12345'
        """
        try:
            # 파일명에서 확장자 제거
            name = str(filename).split('.')[0]  # '0_12345'
            # 언더스코어 기준으로 분리하여 뒷부분 반환
            unicode_value = name.split('_')[1]  # '12345'
            return unicode_value
        except Exception as e:
            raise ValueError(f"Invalid filename format: {filename}. Expected format: number_unicode.png")
    
    def _create_combined_image(self, gothic: Image.Image, handwriting: Image.Image) -> Image.Image:
        """고딕체와 손글씨를 결합하여 새 이미지 생성"""
        # 새 이미지 생성 (128x256, 흰색 배경)
        combined = Image.new('L', (self.img_size * 2, self.img_size), 255)
        
        # 고딕체 붙이기 (왼쪽)
        combined.paste(gothic, (0, 0))
        
        # 손글씨 중앙 정렬을 위한 처리
        hw_array = np.array(handwriting)
        # 흰색이 아닌 픽셀의 범위 찾기
        coords = np.where(hw_array < 255)
        if len(coords[0]) > 0:  # 글자가 있는 경우
            top, bottom = coords[0].min(), coords[0].max()
            left, right = coords[1].min(), coords[1].max()
            
            # 글자 영역 추출
            content = handwriting.crop((left, top, right + 1, bottom + 1))
            
            # 새로운 크기 계산 (비율 유지)
            content_width = right - left + 1
            content_height = bottom - top + 1
            ratio = min(self.img_size * 0.8 / content_width, 
                       self.img_size * 0.8 / content_height)
            
            new_width = int(content_width * ratio)
            new_height = int(content_height * ratio)
            
            # 크기 조정
            content = content.resize((new_width, new_height), Image.LANCZOS)
            
            # 중앙 위치 계산
            paste_x = self.img_size + (self.img_size - new_width) // 2
            paste_y = (self.img_size - new_height) // 2
            
            # 손글씨 붙이기 (오른쪽 중앙)
            combined.paste(content, (paste_x, paste_y))
        
        return combined


def main():
    # 경로 설정 (사용자가 지정할 수 있도록)
    input_dir = "./handwriting_templates"
    output_dir = "./handwritten_result"
    font_id = 10  # 고정된 폰트 ID
    
    # 전처리기 생성 및 실행
    preprocessor = HandwritingPreprocessor(input_dir, output_dir, font_id)
    
    try:
        preprocessor.process_images()
        print("\n이미지 처리가 완료되었습니다!")
        print(f"처리된 이미지가 {output_dir}에 저장되었습니다.")
    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
