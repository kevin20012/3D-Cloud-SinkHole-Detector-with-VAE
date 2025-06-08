import os
import glob

def process_pcd_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 헤더 정보 끝나는 라인 인덱스 찾기
    header_end_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == 'DATA ascii':
            header_end_idx = i
            break

    header = lines[:header_end_idx + 1]
    data_lines = lines[header_end_idx + 1:]

    filtered_data = []
    for line in data_lines:
        if line.strip() == '':
            continue
        parts = line.strip().split()
        label = int(parts[4])  # label은 6번째 컬럼 (index 5)
        if label != 2:
            filtered_data.append(line)

    # 필터링 후 points 개수 업데이트 (헤더에서 POINTS 줄 수정)
    new_points_count = len(filtered_data)
    new_header = []
    for line in header:
        if line.startswith('POINTS'):
            new_header.append(f'POINTS {new_points_count}\n')
        else:
            new_header.append(line)

    # 파일 덮어쓰기
    with open(filepath, 'w') as f:
        f.writelines(new_header)
        f.writelines(filtered_data)


def process_folder(folder_path):
    pcd_files = glob.glob(os.path.join(folder_path, '*.pcd'))
    print(f"Found {len(pcd_files)} pcd files.")

    for file in pcd_files:
        print(f"Processing {file} ...")
        process_pcd_file(file)
    print("Done.")

# 예시 사용법
folder_path = './data'  # 수정 필요
process_folder(folder_path)
