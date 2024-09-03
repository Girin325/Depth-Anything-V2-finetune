from PIL import Image
import numpy as np
import os
import glob

# 입력 및 출력 폴더 경로 설정
input_folder = "/home/san/d_drive/Depth-Anything-V2/custom/val"  # 입력 이미지 폴더 경로
output_folder = "/home/san/d_drive/Depth-Anything-V2/custom/val_crop"  # 출력 이미지 폴더 경로

# 출력 폴더가 존재하지 않으면 생성
os.makedirs(output_folder, exist_ok=True)

# 입력 폴더 내 모든 PNG 파일에 대해 반복
for input_path in glob.glob(os.path.join(input_folder, "*.png")):
    # 이미지 로드
    image = Image.open(input_path).convert("RGB")
    image_np = np.array(image)

    # 사각형 마스크 생성
    h, w = image_np.shape[:2]
    border = 65  # 가장자리의 두께 설정

    # 크롭할 경계 상자 결정
    left = border
    top = border
    right = w - border
    bottom = h - border

    # 이미지를 경계 상자로 크롭
    cropped_image = image.crop((left, top, right, bottom))

    # 출력 파일 경로 생성
    output_path = os.path.join(output_folder, f"crop_{os.path.basename(input_path)}")

    # 크롭한 이미지 저장
    cropped_image.save(output_path)

    print(f"Cropped image saved to {output_path}")

print("All images have been processed and saved.")
