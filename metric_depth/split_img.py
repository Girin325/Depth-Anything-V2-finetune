import os
import shutil
import glob

# 입력 폴더 및 출력 폴더 경로 설정
input_folder = "/home/san/d_drive/shared/train"  # 이미지가 있는 입력 폴더 경로
output_folder = "/home/san/d_drive/shared/val"  # 4번째마다 이미지를 이동시킬 출력 폴더 경로

# 출력 폴더가 존재하지 않으면 생성
os.makedirs(output_folder, exist_ok=True)

# 입력 폴더 내의 모든 이미지 파일 경로 가져오기
image_files = sorted(glob.glob(os.path.join(input_folder, "*.*")))  # 모든 파일 유형을 포함

# 이미지 파일들을 순서대로 처리
for index, file_path in enumerate(image_files, start=1):
    if index % 5 == 0:  # 4번째 이미지마다 처리
        # 파일명을 추출하고 새로운 출력 경로를 설정
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(output_folder, file_name)

        # 파일을 새로운 경로로 이동
        shutil.move(file_path, destination_path)

        print(f"Moved {file_name} to {output_folder}")

print("All applicable images have been moved.")
