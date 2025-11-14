import pdb
import glob
import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

from e2p import E2P
from pd import PD
from ppi import PPI

FOV = 30

# ToDo: 透視投影画像生成するマップは１回の計算にできそう
# 透視投影画像群の生成
def generate_front_ppis(img_e, fov=FOV, overlap=0.5):
    # 使用する全方位カメラの解像度
    src_size_h = img_e.shape[0]
    src_size_w = img_e.shape[1]
    # 表示する透視投影画像の解像度
    dst_width = int(2.0 * np.tan(np.deg2rad(fov) / 2.0) * src_size_w / (2*np.pi))
    dst_height = int(2.0 * np.tan(np.deg2rad(fov) / 2.0) * src_size_h / np.pi)

    view = E2P(src_size_w, src_size_h, dst_width, dst_height)
    
    THETA_RANGE = 90
    PHI_RANGE = 60
    angle_step = fov * (1 - overlap)
    first_step = fov/2
    theta_eyes = np.arange(-THETA_RANGE + first_step, THETA_RANGE, angle_step)
    phi_eyes = np.arange(PHI_RANGE - first_step, -PHI_RANGE, -angle_step)
    # 透視投影画像を生成する視線角度の設定

    ppis = []

    for j, phi_eye in enumerate(phi_eyes):
        for i, theta_eye in enumerate(theta_eyes):
            # 透視投影画像の生成
            view.generate_map(np.deg2rad(theta_eye), np.deg2rad(phi_eye), 0, 1)
            dst_img = view.generate_image(img_e)
            # 保存
            ppi = PPI(img_e, dst_img, theta_eye, phi_eye)
            ppis.append(ppi)
    return ppis

# 透視投影画像を生成
def generate_ppi(img_e, theta_eye, phi_eye, scale=1, fov=FOV):
    img_e_h = img_e.shape[0]
    img_e_w = img_e.shape[1]
    # 透視投影画像の解像度
    dst_width = int(2.0 * np.tan(np.deg2rad(fov) / 2.0) * img_e_w / (2*np.pi))
    dst_height = int(2.0 * np.tan(np.deg2rad(fov) / 2.0) * img_e_h / np.pi)
    # 透視投影画像の生成．
    view = E2P(img_e_w, img_e_h, dst_width, dst_height)
    view.generate_map(np.deg2rad(theta_eye), np.deg2rad(phi_eye), 0, scale)
    ppi = view.generate_image(img_e)
    return PPI(img_e, ppi, theta_eye, phi_eye)

def calc_optimal_fov_from_scale(ppi, scale):
    focal_length = ppi.get_focal_length()
    w = ppi.get_ppi_w()
    s_p = scale
    fov_rad = np.arctan2(w, (focal_length*s_p))
    fov_deg = np.rad2deg(fov_rad)
    return fov_deg

def collect_gaze_point_candidates(ppis, gaze_point_num=0):
    gaze_point_candidates = []
    for ppi in ppis:
        pose_detector_for_ppi = PD(ppi.get_ppi())
        if pose_detector_for_ppi.is_pose_detected():
            # 注視点（鼻）の座標を取得
            landmark_coordinate_x, landmark_coordinate_y = pose_detector_for_ppi.get_landmark_coordinate(gaze_point_num)
            # 注視点の全方位画像上の角度座標を取得
            theta_e, phi_e = ppi.get_angular_coordinate(landmark_coordinate_x, landmark_coordinate_y)
            gaze_point_candidates.append([theta_e, phi_e])
    return np.array(gaze_point_candidates, dtype=float)

def grouping_points(points, eps=5, min_samples=1):
    if not isinstance(points, np.ndarray):
        raise TypeError("pointsがNumPy配列ではありません。np.ndarrayを渡してください。")
    if points.size == 0:
        return None
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # グルーピングしたラベル配列を出力。例：[0, 0, 1, 1, -1]（ラベルが -1 の場合はノイズ）
    labels = dbscan.fit_predict(points)
    # 点をグループ分けした辞書を作成。例：｛　0: array([[x1, y1], [x2, y2]])　｝
    grouped_points = {}
    for label in set(labels):
        grouped_points[label] = points[labels == label]
    return grouped_points

def centering_points(points):
    if points.shape[0] == 0:
        return None
    centered_point = np.mean(points, axis=0)
    return centered_point

# 人が透視投影画像に1/kの高さで写る。例： k=2のとき、人が透視投影画像の1/2の高さになる
def scaling_person_by_height(ppi, k=2):
    pose_detector_for_adjusted = PD(ppi.get_ppi())
    if pose_detector_for_adjusted.is_pose_detected():
        minXY, maxXY = pose_detector_for_adjusted.get_boundingbox_coordinates()
        H_b = maxXY[1] - minXY[1]
        H_i = ppi.get_ppi_h()
        ppi_scale = H_i/(k * H_b)
        scaling_fov = calc_optimal_fov_from_scale(ppi, ppi_scale)
        scaled_ppi = generate_ppi(  ppi.get_src_img(), 
                                    ppi.get_angle_u(),
                                    ppi.get_angle_v(),
                                    fov= scaling_fov)
        return scaled_ppi
    
# 人が透視投影画像に1/kの面積で写る。例： k=2のとき、人が透視投影画像の1/2の面積になる
def scaling_person_by_surface(ppi, k=2):
    pose_detector_for_adjusted = PD(ppi.get_ppi())
    if pose_detector_for_adjusted.is_pose_detected():
        bb_S = pose_detector_for_adjusted.get_boudingbox_surface()
        img_ratio = ppi.get_ppi_w() / ppi.get_ppi_h()
        W_dash = np.sqrt(bb_S * img_ratio * k)
        img_w = ppi.get_ppi_w()
        ppi_scale = img_w / W_dash
        scaled_ppi = generate_ppi(  ppi.get_src_img(), 
                                    ppi.get_angle_u(),
                                    ppi.get_angle_v(),
                                    ppi_scale)
        return scaled_ppi

def generate_bullettime(img):
    ppis = generate_front_ppis(img)
    collect_gaze_point_candidate = collect_gaze_point_candidates(ppis)
    if collect_gaze_point_candidate.size == 0:
        return None
    grouped_points = grouping_points(collect_gaze_point_candidate)
    centered_points = list(map(centering_points, grouped_points.values()))
    gaze_ppis = [ generate_ppi(img, cp[0], cp[1]-90) for cp in centered_points ]
    scaled_ppis = list(filter(None, map(scaling_person_by_height, gaze_ppis)))
    return [ppi.get_ppi() for ppi in scaled_ppis]

def generate_bullettime_dubug(output_path, file_name_pattern, img):
    ppis = generate_front_ppis(img)

    # 骨格検出結果
    for ppi in ppis:
        pose_detector_for_ppi = PD(ppi.get_ppi())
        if pose_detector_for_ppi.is_pose_detected():
            ppi_with_landmarks = pose_detector_for_ppi.draw_pose_landmarks()
            save_imgs(output_path, f"{file_name_pattern}_0_pd_result", ppi_with_landmarks)

    collect_gaze_point_candidate = collect_gaze_point_candidates(ppis)
    if collect_gaze_point_candidate.size == 0:
        return None
    grouped_points = grouping_points(collect_gaze_point_candidate)
    centered_points = list(map(centering_points, grouped_points.values()))
    gaze_ppis = [ generate_ppi(img, cp[0], cp[1]-90) for cp in centered_points ]

    # スケール前
    for ppi in gaze_ppis:
        pose_detector_for_ppi = PD(ppi.get_ppi())
        if pose_detector_for_ppi.is_pose_detected():
            ppi_with_landmarks = pose_detector_for_ppi.draw_pose_landmarks()
            save_imgs(output_path, f"{file_name_pattern}_1_before_scale", ppi_with_landmarks)

    scaled_ppis = list(filter(None, map(scaling_person_by_height, gaze_ppis)))

    # スケール後
    for ppi in scaled_ppis:
        ppi_img = ppi.get_ppi()
        save_imgs(output_path, f"{file_name_pattern}_2_after_scale", ppi_img)


# バレットタイム画像を作成
# def generate_bullettime(imgs):
#     bullettime = []
#     count = 0
#     for img in imgs:
#         ppis = generate_front_ppis(img)
#         gaze_point_candidates = collect_gaze_point_candidates(ppis)
#         if gaze_point_candidates.size == 0:
#             break
#         grouped_points = grouping_points(gaze_point_candidates)
#         for gaze_point_candidate in grouped_points.values():
#             # 注視点を決定  
#             gaze_point = centering_points(gaze_point_candidate)
#             # 注視点の透視投影画像を作成            
#             theta_eye = gaze_point[0]
#             phi_eye = gaze_point[1]-90 # 角度座標に変換
#             grouped_ppi = generate_ppi(img, theta_eye, phi_eye)
#             count += 1
#             save_img(output_path, f"img_{count}.jpg",grouped_ppi.get_ppi())
#             # スケーリング
#             scaled_ppi = scaling_person_by_height(grouped_ppi)
#             if scaled_ppi == None:
#                 break
#             bullettime.append(scaled_ppi.get_ppi())
#     return bullettime

# Todo: ファイル命名の修正。例：01_02.mp4_0002.jpg_0001.jpg
def save_imgs(output_path, file_name_pattern, imgs, expand=".jpg"):
    os.makedirs(output_path, exist_ok=True)
    for i, img in enumerate(imgs):
        file_name = file_name_pattern.format(f"{i+1:04d}") + expand
        file_path = os.path.join(output_path, file_name)
        cv2.imwrite(file_path, img)
        print(f"{file_path} を保存しました。")

if __name__ == '__main__':
    input_path = "../input/tennis_center"
    output_path = "../output/tennis_center_bullettime_opt_fov"
    img_paths = glob.glob(os.path.join(input_path, "*.jpg")) 

    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        imgs.append(img)
        filename_with_ext = os.path.basename(img_path)
        filename_without_ext = os.path.splitext(filename_with_ext)[0]
        file_name_pattern = filename_without_ext
        bullettime = generate_bullettime_dubug(output_path, file_name_pattern, img)