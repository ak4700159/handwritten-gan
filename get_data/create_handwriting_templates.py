import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_handwriting_templates(characters, output_dir, font_path="./fonts/source/source_font.ttf"):
    """
    고딕체 글자와 손글씨 공간이 있는 템플릿 이미지들을 생성
    
    Args:
        characters (list): 생성할 한글 글자 리스트
        output_dir (str): 결과물을 저장할 디렉토리 경로
        font_path (str): 고딕체 폰트 파일 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 선명한 텍스트를 위해 실제 출력 크기보다 2배 큰 이미지에 렌더링
    scale_factor = 2
    target_size = (256, 128)
    scaled_size = (target_size[0] * scale_factor, target_size[1] * scale_factor)
    
    # 폰트 설정 (고딕체) - 스케일에 맞춰 크기 조정
    font_size = 80 * scale_factor
    font = ImageFont.truetype(font_path, font_size)
    
    for char in characters:
        # 고해상도 이미지 생성
        image = Image.new('RGB', scaled_size, 'white')
        draw = ImageDraw.Draw(image)

                # 왼쪽 영역의 중앙점 계산
        left_center_x = scaled_size[0] // 4
        center_y = scaled_size[1] // 2
        
        # 왼쪽 영역에 고딕체 글자 그리기
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 정확한 중앙 위치 계산
        x = left_center_x - text_width // 2
        y = center_y - text_height // 2
        
        # 글자를 검은색으로 그리기
        draw.text((x, y), char, fill=(0, 0, 0), font=font)
        
        # 오른쪽 영역에 격자 그리기
        mid_x = scaled_size[0]//2
        # # 가로선
        # for i in range(0, scaled_size[1] + 1, 64 * scale_factor):
        #     draw.line([(mid_x, i), (scaled_size[0], i)], 
        #              fill=(200, 200, 200), width=scale_factor)
        # # 세로선
        # for i in range(mid_x, scaled_size[0] + 1, 64 * scale_factor):
        #     draw.line([(i, 0), (i, scaled_size[1])], 
        #              fill=(200, 200, 200), width=scale_factor)
            
        # # 중앙 구분선
        # draw.line([(mid_x, 0), (mid_x, scaled_size[1])], 
        #          fill=(100, 100, 100), width=scale_factor)
        
        # 이미지 크기를 원래 크기로 줄이기 (안티앨리어싱 적용)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 파일 저장 (높은 품질)
        output_path = os.path.join(output_dir, f'0_{ord(char)}.png')
        image.save(output_path, 'PNG', quality=95)

# PDF에서 추출한 글자 리스트
characters = [
    '겅', '겆', '겜', '겯', '궓', '기', '긽', '꺷', '꺾', '껙', '꽁', '꽍', '꾶', '꿀', '꿋',
    '뀀', '뀻', '끎', '끕', '낳', '냄', '냱', '넧', '넨', '놋', '놨', '놰', '뇔', '뇹', '늡',
    '닑', '댽', '뎬', '돎', '됷', '둠', '뒣', '듀', '듬', '땀', '땋', '뗶', '뚦', '뛒', '뛰',
    '뜸', '띔', '랖', '램', '럤', '렝', '련', '렵', '렽', '롯', '롷', '롼', '뢋', '룡', '룩',
    '뤇', '륑', '륨', '륿', '릙', '릞', '링', '맷', '먛', '멈', '멱', '멼', '몴', '뫴', '뭏',
    '뭣', '뭻', '뮴', '바', '밝', '뱝', '벋', '봃', '뵂', '뵐', '뵨', '불', '붻', '뺍', '뺐',
    '뺙', '뽮', '뾏', '쀭', '삘', '삥', '사', '색', '샛', '솎', '솰', '숌', '숭', '싀', '싇',
    '식', '싫', '싱', '싹', '쌤', '쎻', '쏞', '쏫', '쐬', '쑕', '쑹', '씀', '앃', '앍', '압',
    '얌', '얹', '얼', '얽', '엔', '옆', '예', '옫', '왓', '욉', '욧', '워', '웬', '윑', '유',
    '윩', '읓', '잡', '잿', '쟎', '정', '젯', '졎', '존', '죽', '줊', '중', '쥐', '쥿', '즢',
    '즶', '짓', '짘', '짠', '쨋', '쮪', '쯓', '찝', '착', '챤', '챨', '첂', '첺', '쳰', '초',
    '촤', '쵮', '쵕', '축', '춫', '췽', '츯', '컫', '켓', '쾀', '쿰', '퀵', '큭', '탭', '턉',
    '텝', '톕', '톤', '툼', '튠', '트', '틂', '틥', '펨', '퐁', '퐱', '푐', '푻', '퓐', '플',
    '핢', '핥', '험', '혤', '홉', '확', '홴', '횳', '훗', '휊', '휩', '휴', '흥', '흫', '힙'
]

# 템플릿 생성
output_directory = "handwriting_templates"
create_handwriting_templates(characters, output_directory)