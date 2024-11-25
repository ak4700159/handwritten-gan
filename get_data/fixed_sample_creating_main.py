import torch
from dataset import save_fixed_sample

# sample_size(batch_size), img_size, data_dir, save_dir, val=False, verbose=True, with_charid=True, resize_fix=90
def main():
    save_fixed_sample(16, 128, "./dataset", "./fixed_dir")

if __name__ == "__main__":
        # CUDA 사용 가능 여부 확인 및 출력
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    main()