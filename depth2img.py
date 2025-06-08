# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import cv2
import os

def pointcloud_to_depth_image(
    pcd_path: str,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    resolution: float = 0.01,
    visualize: bool = False,
    save_path: str = "depth_map.png"
):
    """
    3D 포인트 클라우드를 로드하여,
    1) RANSAC으로 도로 평면(기준평면) 추출
    2) 평면법선 기준 depth 계산
    3) 평면 기준 UV 좌표 정의 후 2D 격자로 매핑
    4) Depth Map 생성 후 8비트 그레이스케일 이미지 저장
    """

    pcd = o3d.io.read_point_cloud(pcd_path)
    print("[단계 1] 포인트 클라우드 로드 완료 → 총 포인트 수:", np.asarray(pcd.points).shape[0])

    # 평면 추출 (RANSAC)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    a, b, c, d = plane_model
    print("[단계 2] 평면 모델:", plane_model)
    print("[단계 2] 평면에 포함된 점 수(Inliers):", len(inliers))

    # 평면법선 벡터 단위벡터화
    normal = np.array([a, b, c], dtype=np.float64)
    norm_factor = np.linalg.norm(normal)
    normal = normal / norm_factor
    d_normed = d / norm_factor

    # 평면 위 중심 (Centroid, 평균) 계산
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_pts = np.asarray(inlier_cloud.points)
    centroid_plane = inlier_pts.mean(axis=0)  # (3,)

    # 평면 기준 좌표계 (u, v) 정의
    # up 벡터 선정
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(up, normal)) > 0.99:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # u, v 계산 (직교 단위벡터)
    u = np.cross(up, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # 전체 포인트 배열
    pts = np.asarray(pcd.points)  # shape = (N, 3)

    # signed distance (평면으로부터 부호 거리) 계산
    signed_dist = pts.dot(normal) + d_normed  # shape = (N,)
    # depth 정의 (평면 아래로 내려간 부분만 양수)
    depth = np.maximum(0.0, -signed_dist)  # shape = (N,)

    # 평면 기준 상대 좌표 (pts_rel)
    pts_rel = pts - centroid_plane  # shape = (N, 3)
    uv_u = pts_rel.dot(u)  # shape = (N,)
    uv_v = pts_rel.dot(v)  # shape = (N,)

    # 격자 해상도(resolution) → 이미지 크기 결정
    u_min, u_max = uv_u.min(), uv_u.max()
    v_min, v_max = uv_v.min(), uv_v.max()
    W = int(np.ceil((u_max - u_min) / resolution)) + 1
    H = int(np.ceil((v_max - v_min) / resolution)) + 1
    print(f"[단계 7] 해상도(res) = {resolution} m/pixel → 이미지 크기: (W, H) = ({W}, {H})")

    # 빈 depth map 초기화
    depth_map = np.zeros((H, W), dtype=np.float32)

    # 포인트별 픽셀 좌표(i, j) 계산
    i_coords = np.floor((uv_u - u_min) / resolution).astype(np.int32)
    j_coords = np.floor((uv_v - v_min) / resolution).astype(np.int32)

    # 픽셀별 Depth 값 채우기 (최댓값)
    for idx in range(pts.shape[0]):
        if depth[idx] > 0.0:
            x_pix = i_coords[idx]
            y_pix = j_coords[idx]
            # 경계 확인
            if 0 <= x_pix < W and 0 <= y_pix < H:
                if depth[idx] > depth_map[y_pix, x_pix]:
                    depth_map[y_pix, x_pix] = depth[idx]

    # 0~255 정규화 (8비트 그레이스케일)
    depth_max = depth_map.max()
    if depth_max > 0:
        depth_img = (depth_map / depth_max * 255.0).astype(np.uint8)
    else:
        depth_img = depth_map.astype(np.uint8)
    depth_img = cv2.dilate(depth_img, np.ones((5, 5), np.uint8), iterations=3)  # 팽창 처리
    # depth_img = cv2.GaussianBlur(depth_img, (5, 5), 0)  # 가우시안 블러링
    # 시각화
    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.title("Depth Map (도로 결함)")
        plt.imshow(depth_img, cmap='gray')
        plt.axis('off')
        plt.show()

    # 결과 저장
    depth_img = cv2.resize(depth_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    if save_path is not None:
        cv2.imwrite(save_path, depth_img)
        print(f"[단계 13] 깊이 이미지 저장 완료 → '{save_path}'")

    return depth_img, depth_map


if __name__ == "__main__":
    path_dir = './data'
    save_dir = './data2img'
    for f in os.listdir(path_dir):
        file_path = os.path.join(path_dir, f)
        if file_path.endswith('.pcd'):
            print(f"Processing file: {file_path}")
            file_name = f'{f[:-4]}_depth_map.png'
            depth_img, depth_map = pointcloud_to_depth_image(
                pcd_path=file_path,
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=1000,
                resolution=0.0001,
                visualize=True,
                save_path=os.path.join(save_dir, file_name)
            )

