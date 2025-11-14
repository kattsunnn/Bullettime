import pdb
import glob
import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import random
#
import e2p
from pd import PD
from ppi import PPI

from typing import List
from dataclasses import dataclass, field

# 透視投影画像群の生成
def generate_ppis(img_e):
    # 使用する全方位カメラの解像度
    src_size_h = img_e.shape[0]
    src_size_w = img_e.shape[1]
    # 表示する透視投影画像の解像度
    dst_width = int(2.0 * np.tan(np.deg2rad(60) / 2.0) * src_size_w / (2*np.pi))
    dst_height = int(2.0 * np.tan(np.deg2rad(60) / 2.0) * src_size_h / np.pi)
    # 視線角度の設定
    theta_eyes = [-60, -30, 0, 30, 60]
    phi_eyes = [30, 0, -30]

    ppis = []

    for j, phi_eye in enumerate(phi_eyes):
        for i, theta_eye in enumerate(theta_eyes):
            view = e2p.E2P(src_size_w, src_size_h, dst_width, dst_height)
            view.generate_map(np.deg2rad(theta_eye), np.deg2rad(phi_eye), 0, 1)
            dst_img = view.generate_image(img_e)
            ppi = PPI(img_e, dst_img, theta_eye, phi_eye)
            ppis.append(ppi)
    return ppis

# 透視投影画像を生成
def generate_ppi(img_e, angular_coodinate, scale=1):
    img_e_h = img_e.shape[0]
    img_e_w = img_e.shape[1]
    theta_eye = angular_coodinate[0]
    phi_eye = angular_coodinate[1]-90
    # 透視投影画像の解像度
    dst_width = int(2.0 * np.tan(np.deg2rad(60) / 2.0) * img_e_w / (2*np.pi))
    dst_height = int(2.0 * np.tan(np.deg2rad(60) / 2.0) * img_e_h / np.pi)
    # 透視投影画像の生成．
    view = e2p.E2P(img_e_w, img_e_h, dst_width, dst_height)
    view.generate_map(np.deg2rad(theta_eye), np.deg2rad(phi_eye), 0, scale)
    ppi = view.generate_image(img_e)
    return PPI(img_e, ppi, theta_eye, phi_eye)

def get_human_scale(img_S, bb_S, k):
    s =  k * bb_S / img_S
    return s

def show_img(img):
    title = f"image"
    cv2.imshow(title, img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

@dataclass
class ImgInfo:
    img: np.ndarray
    path: str
    save_folder: str
    # ppis: list[PPI] = field(default_factory=list)
    # person_ppis: list[PPI] = field(default_factory=list)

# グループ分けした点をを取得
def get_grouped_person_points(points, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    grouped_points = {}
    for label in set(labels):
        # 各クラスタの点を抽出（ラベルが -1 の場合はノイズ）
        grouped_points[label] = points[labels == label]
    return grouped_points

# グループの中心座標を取得
def get_centered_person_points(points, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    grouped_points = {}
    for label in set(labels):
        # 各クラスタの点を抽出（ラベルが -1 の場合はノイズ）
        grouped_points[label] = points[labels == label]
    centered_points = {}
    for label, group in grouped_points.items():
        if label != -1: 
            centered_point = np.mean(group, axis=0)
            centered_points[label] = centered_point
    return centered_points

# 姿勢を用いた人物対応付け
def find_most_similar_pose(reference_pose, candidate_poses):
    # 点のユークリッド距離を計算
    def pose_distance(pose1, pose2):
        assert pose1.shape == pose2.shape, "キーポイントの数が一致しません"
        dists = np.linalg.norm(pose1 - pose2, axis=1)
        return np.mean(dists)
    
    distances = [pose_distance(reference_pose, candidate) for candidate in candidate_poses]
    best_idx = int(np.argmin(distances))
    return distances, best_idx
    # return distances

# 人物ごとの注視画像を生成
def generate_bullettimeimg(img_info):
    for cam_num, img in enumerate(img_info):
        src_img = img.img
        # 透視投影画像群の生成
        ppis = generate_ppis(src_img)
        # img_info.ppis = ppis
        angular_coordinate_list = []
        for ppi_num, ppi in enumerate(ppis):
            pose_detector_for_ppi = PD(ppi.get_ppi())
            if pose_detector_for_ppi.is_pose_detected():
                # 注視点の座標を取得
                landmark_coordinate_x, landmark_coordinate_y = pose_detector_for_ppi.get_landmark_coordinate()
                # 注視点の角度座標を取得
                theta_e, phi_e = ppi.get_angular_coordinate(landmark_coordinate_x, landmark_coordinate_y)
                angular_coordinate_list.append([theta_e, phi_e])
        angular_coordinate_array = np.array(angular_coordinate_list)
        person_points = get_centered_person_points(angular_coordinate_array, eps=5, min_samples=1)
        #人物ごとの注視画像を生成
        for label, gaze_point in person_points.items():
            #注視画像の生成（スケーリングなし）
            adjusted_ppi = generate_ppi(src_img, gaze_point)
            # スケールを取得
            pose_detector_for_adjusted = PD(adjusted_ppi.get_ppi())
            if pose_detector_for_adjusted.is_pose_detected():
                # 高さでスケーリング
                minXY, maxXY = pose_detector_for_adjusted.get_boundingbox_coordinates()
                H_b = maxXY[1] - minXY[1]
                H_i = adjusted_ppi.get_ppi_h()
                k = 2.5
                ppi_scale = H_i/(k * H_b) 
                #面積でスケール
                # bb_S = pose_detector_for_adjusted.get_boudingbox_surface()
                # img_ratio = adjusted_ppi.get_ppi_w() / adjusted_ppi.get_ppi_h()
                # k = 10
                # W_dash = np.sqrt(bb_S * img_ratio * k)
                # img_w = adjusted_ppi.get_ppi_w()
                # ppi_scale = img_w / W_dash
            # スケーリングした注視画像の生成
            scaled_ppi = generate_ppi(src_img, gaze_point, ppi_scale)
            #画像の保存
            # drawed_ppi = pose_detector_for_ppi.draw_pose_gazepoint_bb(pose_detector_for_ppi.get_landmark_coordinate())
            # save_path_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_ppi_{ppi_num+1}.jpg")
            # cv2.imwrite(save_path_ppi, ppi.img_p)
            # save_path_adjusted_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_adjusted_ppi_{ppi_num+1}.jpg")
            # cv2.imwrite(save_path_adjusted_ppi, adjusted_ppi.img_p)
            # save_path_drawed_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_drawed_ppi_{ppi_num+1}.jpg")
            # cv2.imwrite(save_path_drawed_ppi, drawed_ppi)
            save_path_scaled_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_scaled_ppi_{label+1}.jpg")
            cv2.imwrite(save_path_scaled_ppi, scaled_ppi.img_p)
# 人物ごとのBB画像を保存
def save_human_cropped_img(img_info):
    for cam_num, img in enumerate(img_info):
        src_img = img.img
        # 透視投影画像群の生成
        ppis = generate_ppis(src_img)
        # img_info.ppis = ppis
        angular_coordinate_list = []
        for ppi_num, ppi in enumerate(ppis):
            pose_detector_for_ppi = PD(ppi.get_ppi())
            if pose_detector_for_ppi.is_pose_detected():
                # 注視点の座標を取得
                landmark_coordinate_x, landmark_coordinate_y = pose_detector_for_ppi.get_landmark_coordinate()
                # 注視点の角度座標を取得
                theta_e, phi_e = ppi.get_angular_coordinate(landmark_coordinate_x, landmark_coordinate_y)
                angular_coordinate_list.append([theta_e, phi_e])
        angular_coordinate_array = np.array(angular_coordinate_list)
        person_points = get_centered_person_points(angular_coordinate_array, eps=5, min_samples=1)
        #人物ごとの注視画像を生成
        for label, gaze_point in person_points.items():
            #注視画像の生成（スケーリングなし）
            adjusted_ppi = generate_ppi(src_img, gaze_point)
            # スケールを取得
            pose_detector_for_adjusted = PD(adjusted_ppi.get_ppi())
            if pose_detector_for_adjusted.is_pose_detected():
                # 高さでスケーリング
                minXY, maxXY = pose_detector_for_adjusted.get_boundingbox_coordinates()
                H_b = maxXY[1] - minXY[1]
                H_i = adjusted_ppi.get_ppi_h()
                k = 2.5
                ppi_scale = H_i/(k * H_b) 
                #面積でスケール
                # bb_S = pose_detector_for_adjusted.get_boudingbox_surface()
                # img_ratio = adjusted_ppi.get_ppi_w() / adjusted_ppi.get_ppi_h()
                # k = 10
                # W_dash = np.sqrt(bb_S * img_ratio * k)
                # img_w = adjusted_ppi.get_ppi_w()
                # ppi_scale = img_w / W_dash
            # スケーリングした注視画像の生成
            scaled_ppi = generate_ppi(src_img, gaze_point, ppi_scale)
            pose_detector_for_scaled = PD(scaled_ppi.get_ppi())
            if pose_detector_for_scaled.is_pose_detected():
                cropped_img = pose_detector_for_scaled.crop_boundingbox()
            #画像の保存
                save_path_cropped_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_cropped_ppi_{label+1}.jpg")
                cv2.imwrite(save_path_cropped_ppi, cropped_img)

# 特徴点マッチングを用いた人物の対応付け
def human_matching_by_featurepoint():
    # 特徴点を用いた人物対応付け
    def search_similar_img(query_img, candidate_imgs, ratio_thresh=0.75, match_thresh=5):
        # カラー画像かチェック
        if query_img is None or len(query_img.shape) != 3:
            raise ValueError("query_img must be a valid BGRカラー画像 (3チャンネル)")
        # 特徴抽出器やマッチングアルゴリズムの設定
        # feature_extractor = cv2.SIFT_create()
        feature_extractor = cv2.AKAZE_create(threshold=0.0005)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        best_score = -1
        best_img = None
        best_idx = None
        # 参照画像の特徴抽出
        query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        kp_query, des_query = feature_extractor.detectAndCompute(query_gray, None)
        if des_query is None or len(kp_query) < 2:
            return None
        # 候補画像群と比較
        for idx, candidate_img in enumerate(candidate_imgs):
            if candidate_img is None or len(img.shape) != 3:
                continue  # カラー画像でなければスキップ
            # 候補画像の特徴抽出
            candidate_img_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
            kp, des = feature_extractor.detectAndCompute(candidate_img_gray, None)
            if des is None or len(kp) < 2:
                continue
            # 特徴点マッチング
            matches = matcher.knnMatch(des_query, des, k=2)
            good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
            match_score = len(good_matches)
            # 可視化
            for i, match in enumerate(good_matches):
                print(f"Match {i}: Distance = {match.distance:.4f}")
            print(f"Group{idx}: {match_score}")
            match_img = cv2.drawMatches(
                query_img, kp_query, candidate_img, kp, good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            # 保存
            save_dir = "outputs/matches"
            os.makedirs(save_dir, exist_ok=True)  # 保存先ディレクトリを作成（存在しない場合）
            save_path = os.path.join(save_dir, f"match_result_{cam_name}_{idx}.png")
            cv2.imwrite(save_path, match_img)
            # 類似画像の更新
            if match_score > best_score and match_score >= match_thresh:
                best_score = match_score
                best_img = candidate_img  # カラー画像を保持
                best_idx = idx

        return best_img, best_idx, best_score
    # ColorHistgramを用いた人物の対応付け
    def search_similar_img_by_sift_and_colorhist(query_img, candidate_imgs, ratio_thresh=0.75, match_thresh=0.1):
        # カラー画像チェック
        if query_img is None or len(query_img.shape) != 3:
            raise ValueError("query_img must be a valid BGRカラー画像 (3チャンネル)")
        # 特徴抽出器と特徴マッチング方法を選択
        feature_extractor = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        sift_weight = 0
        chist_wight = 1
        # 参照画像の特徴抽出
        query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)        
        kp_query, des_query = feature_extractor.detectAndCompute(query_gray, None)
        # desが有効か確認
        if des_query is None or len(kp_query) < 2:
            return None
        # 参照画像の色ヒストグラム
        query_hist = cv2.calcHist([query_img], [0, 1, 2], None, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])
        query_hist_norm = cv2.normalize(query_hist, query_hist).flatten() # スケールの影響を軽減するため正規化

        best_score = -1
        best_img = None
        best_idx = None

        for idx, cand_img in enumerate(candidate_imgs):
            # カラー画像チェック
            if cand_img is None or len(img.shape) != 3:
                continue  
            # 候補画像の特徴抽出
            cand_gray = cv2.cvtColor(cand_img, cv2.COLOR_BGR2GRAY)
            kp_cand, des_cand = feature_extractor.detectAndCompute(cand_gray, None)
            if des_cand is None or len(kp_cand) < 2:
                continue
            # マッチング
            matches = matcher.knnMatch(des_query, des_cand, k=2)
            good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
            match_score = len(good_matches)
            # 候補画像の色ヒストグラム
            cand_hist = cv2.calcHist([cand_img], [0, 1, 2], None, [8, 8, 8],
                                    [0, 256, 0, 256, 0, 256])
            cand_hist_norm = cv2.normalize(cand_hist, cand_hist).flatten()
            hist_score = cv2.compareHist(query_hist_norm, cand_hist_norm, cv2.HISTCMP_CORREL)
            # スコアの合成（SIFTのマッチ数とヒストグラムの類似度）
            total_score = (match_score * sift_weight) + (hist_score * chist_wight)  

            print(f"Group{idx}: {total_score}")
            # 可視化
            match_img = cv2.drawMatches(
                query_img, kp_query, cand_img, kp_cand, good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            # 保存
            save_dir = "outputs/matches"
            os.makedirs(save_dir, exist_ok=True)  # 保存先ディレクトリを作成（存在しない場合）
            save_path = os.path.join(save_dir, f"match_result_{cam_name}_{idx}.png")
            cv2.imwrite(save_path, match_img)

            if total_score > best_score and total_score >= match_thresh:
                best_score = total_score
                best_img = cand_img  # カラー画像を保持
                best_idx = idx

        return best_img, best_idx, best_score

    # 参照画像
    first_reference_path = "outputs/camera_1/camera_1_cropped_ppi_2.jpg"
    query_img = cv2.imread(first_reference_path)
    # 候補画像のフォルダ
    base_dir="outputs"

    for cam_idx in range(1, 10):
        cam_name = f"camera_{cam_idx}"
        cam_dir = os.path.join(base_dir, cam_name)

        candidate_files = sorted([
            f for f in os.listdir(cam_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        candidate_imgs = []
        for fname in candidate_files:
            img_path = os.path.join(cam_dir, fname)
            img = cv2.imread(img_path)
            candidate_imgs.append(img)

        best_img, best_idx, best_score = search_similar_img(query_img, candidate_imgs)

        if best_img is not None:
            print(f"{cam_name}| Best: Group{best_idx} | BestScore: {best_score}")
            # cv2.imshow(f"Best Match in {cam_name}", best_img)  # ← 追加
            # cv2.waitKey(0)  # キー押下まで待機
            # cv2.destroyAllWindows()
            query_img = best_img  # 次の参照画像として更新
        else:
            print("can't find a similar image")

# ランドマークを保存
def save_landmark(img_info):
    for cam_num, img in enumerate(img_info):
        src_img = img.img
        # 透視投影画像群の生成
        ppis = generate_ppis(src_img)
        # img_info.ppis = ppis
        angular_coordinate_list = []
        for ppi_num, ppi in enumerate(ppis):
            pose_detector_for_ppi = PD(ppi.img_p)
            if pose_detector_for_ppi.is_pose_detected():
                # 注視点の座標を取得
                landmark_coordinate_x, landmark_coordinate_y = pose_detector_for_ppi.get_landmark_coordinate()
                # 注視点の角度座標を取得
                theta_e, phi_e = ppi.get_angular_coordinate(landmark_coordinate_x, landmark_coordinate_y)
                angular_coordinate_list.append([theta_e, phi_e])
        angular_coordinate_array = np.array(angular_coordinate_list)
        person_points = get_centered_person_points(angular_coordinate_array, eps=5, min_samples=1)
        #人物ごとの注視画像を生成
        for label, gaze_point in person_points.items():
            #注視画像の生成（スケーリングなし）
            adjusted_ppi = generate_ppi(src_img, gaze_point)
            # スケールを取得
            pose_detector_for_adjusted = PD(adjusted_ppi.get_ppi())
            if pose_detector_for_adjusted.is_pose_detected():
                # 高さでスケーリング
                minXY, maxXY = pose_detector_for_adjusted.get_boundingbox_coordinates()
                H_b = maxXY[1] - minXY[1]
                H_i = adjusted_ppi.get_ppi_h()
                k = 2.5
                ppi_scale = H_i/(k * H_b) 
                #面積でスケール
                # bb_S = pose_detector_for_adjusted.get_boudingbox_surface()
                # img_ratio = adjusted_ppi.get_ppi_w() / adjusted_ppi.get_ppi_h()
                # k = 10
                # W_dash = np.sqrt(bb_S * img_ratio * k)
                # img_w = adjusted_ppi.get_ppi_w()
                # ppi_scale = img_w / W_dash
            # スケーリングした注視画像の生成
            scaled_ppi = generate_ppi(src_img, gaze_point, ppi_scale)
            pose_detector_for_scaled = PD(scaled_ppi.get_ppi())
            if pose_detector_for_scaled.is_pose_detected():
                    normalized_coords = pose_detector_for_scaled.get_normalized_landmark_coordinates()
                    keypoint_save_path = os.path.join(img.save_folder, f"camera_{cam_num+1}_landmark_{label+1}.txt")
                    np.savetxt(keypoint_save_path, normalized_coords, fmt="%.6f", delimiter=",")
            else:
                print(f"cant find camera_{cam_num+1}_landmark_{label+1}.txt")
            #画像の保存
            # drawed_ppi = pose_detector_for_ppi.draw_pose_gazepoint_bb(pose_detector_for_ppi.get_landmark_coordinate())
            # save_path_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_ppi_{ppi_num+1}.jpg")
            # cv2.imwrite(save_path_ppi, ppi.img_p)
            # save_path_adjusted_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_adjusted_ppi_{ppi_num+1}.jpg")
            # cv2.imwrite(save_path_adjusted_ppi, adjusted_ppi.img_p)
            # save_path_drawed_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_drawed_ppi_{ppi_num+1}.jpg")
            # cv2.imwrite(save_path_drawed_ppi, drawed_ppi)
            save_path_scaled_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_scaled_ppi_{label+1}.jpg")
            cv2.imwrite(save_path_scaled_ppi, scaled_ppi.get_ppi())
# ランドマークを用いた人物の対応付け
def human_matching_by_landmark():
    # 基準姿勢を読み込む
    reference_path = "outputs/camera_1/camera_1_landmark_2.txt"
    reference_pose = np.loadtxt(reference_path, delimiter=",")
    base_dir="outputs"
    # camera_2 ～ camera_9 に対して順次比較
    for cam_idx in range(2, 10):
        cam_name = f"camera_{cam_idx}"
        cam_dir = os.path.join(base_dir, cam_name)
    
        candidate_files = sorted([
            f for f in os.listdir(cam_dir)
            if f.endswith(".txt") and "landmark" in f
        ])

        candidate_poses = []
        for file in candidate_files:
            path = os.path.join(cam_dir, file)
            pose = np.loadtxt(path, delimiter=",")
            candidate_poses.append(pose)
        
        dist_list, best_idx = find_most_similar_pose(reference_pose, candidate_poses)
        best_file = candidate_files[best_idx]
        # best_path = os.path.join(cam_dir, best_file)
        summary = "  ".join([f"Group{i}: {d:.4f}" for i, d in enumerate(dist_list)])
        print(f"{cam_name}| Best: Group{best_idx} | Distance: {summary} ")

        # 次の比較に使うreferenceを更新
        reference_pose = candidate_poses[best_idx]

    print("✅ 全てのカメラと比較完了！")

def draw_grouped_person_points_on_omniimg(img_info):
    for cam_num, img in enumerate(img_info):
        src_img = img.img
        src_img_h = src_img.shape[0]
        src_img_w = src_img.shape[1]
        #色
        BRIGHT_COLORS = [
            (0, 0, 255),     # Red
            (255, 0, 0),     # Blue
            (0, 255, 0),     # Green
            (0, 255, 255)    # Yellow
        ]
        # 透視投影画像群の生成
        ppis = generate_ppis(src_img)
        # img_info.ppis = ppis
        angular_coordinate_list = []
        for ppi_num, ppi in enumerate(ppis):
            pose_detector_for_ppi = PD(ppi.get_ppi())
            if pose_detector_for_ppi.is_pose_detected():
                # 注視点の座標を取得
                landmark_coordinate_x, landmark_coordinate_y = pose_detector_for_ppi.get_landmark_coordinate()
                # 注視点の角度座標を取得
                theta_e, phi_e = ppi.get_angular_coordinate(landmark_coordinate_x, landmark_coordinate_y)
                angular_coordinate_list.append([theta_e, phi_e])
        angular_coordinate_array = np.array(angular_coordinate_list)
        grouped_person_points = get_grouped_person_points(angular_coordinate_array, eps=5, min_samples=1)

        for label, points in grouped_person_points.items():
            color = BRIGHT_COLORS[label % len(BRIGHT_COLORS)]  # 4色を循環
            for theta_e_deg, phi_e_deg in points:
                theta_e_rad = np.deg2rad(theta_e_deg)
                phi_e_rad = np.deg2rad(phi_e_deg)
                u = int( (theta_e_rad+np.pi)*(src_img_w/(2*np.pi)) )
                v = int( -(phi_e_rad-np.pi)*(src_img_h/np.pi) )
                print(u, v)
                cv2.circle(src_img, (u, v), radius=5, color=color, thickness=-1)
        save_path_drawed_img_e = os.path.join(img.save_folder, f"camera_{cam_num+1}_drawed_img_e.jpg")
        cv2.imwrite(save_path_drawed_img_e, src_img)

def output_person_points(img_info):
    for cam_num, img in enumerate(img_info):
        src_img = img.img
        src_img_h = src_img.shape[0]
        src_img_w = src_img.shape[1]
        # 透視投影画像群の生成
        ppis = generate_ppis(src_img)
        # img_info.ppis = ppis
        angular_coordinate_list = []
        for ppi_num, ppi in enumerate(ppis):
            pose_detector_for_ppi = PD(ppi.get_ppi())
            if pose_detector_for_ppi.is_pose_detected():
                # 注視点の座標を取得
                landmark_coordinate_x, landmark_coordinate_y = pose_detector_for_ppi.get_landmark_coordinate()
                # 注視点の角度座標を取得
                theta_e, phi_e = ppi.get_angular_coordinate(landmark_coordinate_x, landmark_coordinate_y)
                angular_coordinate_list.append([theta_e, phi_e])
        angular_coordinate_array = np.array(angular_coordinate_list)
        grouped_person_points = get_grouped_person_points(angular_coordinate_array, eps=5, min_samples=1)
        centered_person_points = get_centered_person_points(angular_coordinate_array, eps=5, min_samples=1)

        print(f"camera{cam_num+1}")

        for label, points in grouped_person_points.items():
            for theta_e_deg, phi_e_deg in points:
                theta_e_rad = np.deg2rad(theta_e_deg)
                phi_e_rad = np.deg2rad(phi_e_deg)
                u = int( (theta_e_rad+np.pi)*(src_img_w/(2*np.pi)) )
                v = int( -(phi_e_rad-np.pi)*(src_img_h/np.pi) )
                print(f"grouped:{label}, ({u},{v})")
        
        for label, point in centered_person_points.items():
            theta_e_deg, phi_e_deg = point
            theta_e_rad = np.deg2rad(theta_e_deg)
            phi_e_rad = np.deg2rad(phi_e_deg)
            u = int( (theta_e_rad+np.pi)*(src_img_w/(2*np.pi)) )
            v = int( -(phi_e_rad-np.pi)*(src_img_h/np.pi) )
            print(f"centered:{label}, ({u},{v})")
        pdb.set_trace()

def save_ppis(img_info):
    for cam_num, img in enumerate(img_info):
        src_img = img.img
        src_img_h = src_img.shape[0]
        src_img_w = src_img.shape[1]
        # 透視投影画像群の生成
        ppis = generate_ppis(src_img)
        for ppi_num, ppi in enumerate(ppis):
            pose_detector_for_ppi = PD(ppi.get_ppi())
            if pose_detector_for_ppi.is_pose_detected():
                gaze_point = pose_detector_for_ppi.get_landmark_coordinate()
                drawn_ppi = pose_detector_for_ppi.draw_pose_gazepoint_bb(gaze_point)
                save_path_ppi = os.path.join(img.save_folder, f"camera_{cam_num+1}_ppi{ppi_num+1}.jpg")
                cv2.imwrite(save_path_ppi, drawn_ppi)

# メイン関数
def main():

    # 入力画像のパスを読み込み
    img_folder = "inputs/positionA"
    img_paths = glob.glob(os.path.join(img_folder, "*.jpg")) 

    img_info = []
    output_folder = "outputs"

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"読み込み失敗: {path}")
            continue
        
        # 出力フォルダの生成
        base_name = os.path.splitext(os.path.basename(path))[0]
        save_folder = os.path.join(output_folder, base_name)
        os.makedirs(save_folder, exist_ok=True)

        img_info.append(ImgInfo(img, path, save_folder))

    # test_pose_matching()
    # save_pose_coordinates(img_info)
    # save_human_cropped_img(img_info)
    human_matching_by_featurepoint()

if __name__ == '__main__':
    main()

