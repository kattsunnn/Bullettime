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

# スケールから適切なFOVを計算。
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
        # 解像度を一定に保ちたい場合。
        # scaled_ppi = generate_ppi(  ppi.get_src_img(), 
        #                             ppi.get_angle_u(),
        #                             ppi.get_angle_v(),
        #                             scale=ppi_scale) 
        # スケールに応じて解像度を自動調節
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

# 中間結果を出力する
def generate_bullettime_dubug(output_path, file_name_pattern, img):
    ppis = generate_front_ppis(img)

    # 骨格検出結果
    if ppis == None: 
        return
    pd_result = []
    for ppi in ppis:
        pose_detector_for_ppi = PD(ppi.get_ppi())
        if pose_detector_for_ppi.is_pose_detected():
            ppi_with_landmarks = pose_detector_for_ppi.draw_pose_landmarks()
            pd_result.append(ppi_with_landmarks)
    if pd_result == []:
        return
    save_imgs(output_path, f"{file_name_pattern}_0_pd_result_{{}}", pd_result)
    save_imgs(f"{output_path}/pd_result", f"{file_name_pattern}_{{}}", pd_result)


    collect_gaze_point_candidate = collect_gaze_point_candidates(ppis)
    if collect_gaze_point_candidate.size == 0:
        return None
    grouped_points = grouping_points(collect_gaze_point_candidate)
    centered_points = list(map(centering_points, grouped_points.values()))
    gaze_ppis = [ generate_ppi(img, cp[0], cp[1]-90) for cp in centered_points ]

    # スケール前
    if gaze_ppis == None: 
        return
    before_scale = []
    for ppi in gaze_ppis:
        pose_detector_for_ppi = PD(ppi.get_ppi())
        if pose_detector_for_ppi.is_pose_detected():
            ppi_with_landmarks = pose_detector_for_ppi.draw_pose_landmarks()
            before_scale.append(ppi_with_landmarks)
    if before_scale == []:
        return
    save_imgs(output_path, f"{file_name_pattern}_1_before_scale_{{}}", before_scale)
    save_imgs(f"{output_path}/before_scale", f"{file_name_pattern}_{{}}", before_scale)

    scaled_ppis = list(filter(None, map(scaling_person_by_height, gaze_ppis)))

    # スケール後
    if scaled_ppis == None: 
        return
    after_scale = []
    for ppi in scaled_ppis:
        ppi_img = ppi.get_ppi()
        after_scale.append(ppi_img)
    if after_scale == []:
        return
    save_imgs(output_path, f"{file_name_pattern}_2_after_scale_{{}}", after_scale)
    save_imgs(f"{output_path}/after_scale", f"{file_name_pattern}_{{}}", after_scale)


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

# 仕様：単一画像・複数画像を処理可能
def is_grayscale(imgs):
    # 単一画像をリストに変換
    imgs = [imgs] if isinstance(imgs, np.ndarray) else imgs
    is_gray_img = lambda img: isinstance(img, np.ndarray) and len(img.shape) == 2
    return all(map(is_gray_img, imgs))

def search_most_similar_img_by_sift(query_img_gray, candidate_imgs_gray, ratio_thresh=0.75, match_thresh=5):
        if is_grayscale(query_img_gray):
            TypeError("画像ファイルまたはグレースケールではありません")
        if is_grayscale(candidate_imgs_gray):
            TypeError("画像ファイルまたはグレースケールではありません")
        # 特徴抽出器やマッチングアルゴリズムの設定
        feature_extractor = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        best_score = -1
        best_img = None
        best_idx = None
        # クエリ画像の特徴抽出
        kp_query, des_query = feature_extractor.detectAndCompute(query_img_gray, None)
        if des_query is None or len(kp_query) < 2:
            return None
        # 候補画像群と比較
        for idx, candidate_img in enumerate(candidate_imgs_gray):
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
    return most_similar_img

def search_similar_img_by_sift_and_colorhist(query_img, candidate_imgs, ratio_thresh=0.75, match_thresh=5):
    return most_similar_img


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


def save_imgs(output_path, file_name_pattern, imgs, expand=".jpg"):
    os.makedirs(output_path, exist_ok=True)
    for i, img in enumerate(imgs):
        file_name = file_name_pattern.format(f"{i+1:04d}") + expand
        file_path = os.path.join(output_path, file_name)
        cv2.imwrite(file_path, img)
        print(f"{file_path} を保存しました。")

if __name__ == '__main__':
    input_path = "../input/tennis_serve"
    output_path = f"../output/tennis_serve_bullettime_fov{FOV}"
    img_paths = glob.glob(os.path.join(input_path, "*.jpg")) 

    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        imgs.append(img)
        filename_with_ext = os.path.basename(img_path)
        filename_without_ext = os.path.splitext(filename_with_ext)[0]
        file_name_pattern = filename_without_ext
        bullettime = generate_bullettime_dubug(output_path, file_name_pattern, img)