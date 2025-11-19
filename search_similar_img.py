import numpy as np
import cv2


# 仕様：単一画像・複数画像を処理可能
def is_grayscale(imgs):
    # 単一画像をリストに変換
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    is_gray_img = lambda img: isinstance(img, np.ndarray) and len(img.shape) == 2
    return all(map(is_gray_img, imgs))

def is_colorscale(imgs):
    # 単一画像をリストに変換
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    is_color_img = lambda img: isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3
    return all(map(is_color_img, imgs))

def ensure_grayscale(imgs):
    # 単一画像をリストに変換
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    if is_grayscale(imgs):
        gray_imgs = imgs
    elif is_colorscale(imgs):
        gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    else:
        raise TypeError("サポートしていない画像フォーマットです")

    return gray_imgs if len(gray_imgs)>1 else gray_imgs[0]

def has_enough_keypoints(keypoints, min_points=10):
    if keypoints is None:
        return False
    return len(keypoints) >= min_points

def ratio_test(nearest_distance, second_nearest_distance, ratio_thresh=0.75):
    return nearest_distance < ratio_thresh * second_nearest_distance

def ransac(query_keypoints, candidate_keypoints, matches, ransac_thresh=5.0):
    # Homography 推定には最低4点必要
    if len(matches) < 4:
        return []
    # 対応する座標リストを作成
    src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([candidate_keypoints[m.trainIdx].pt for m in matches])
    # Homography を RANSAC で推定
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]

    return inlier_matches

# Todo: マッチングのアルゴリズムを確認
# Todo: RANSACをする必要があるかも
def search_most_similar_img_by_sift(query_img_gray, candidate_imgs_gray, ratio_thresh=0.75, ransac_thresh=0.5, match_thresh=5):
    if is_grayscale(query_img_gray):
        TypeError("クエリ画像にグレー画像を入力してください")
    if is_grayscale(candidate_imgs_gray):
        TypeError("候補画像にグレー画像を入力してください")
    sift = cv2.SIFT_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # CrossCheckにより、マッチングを1対1に限定
    max_matches = -1
    most_similar_img_idx = None
    most_similar_img = None
    # クエリ画像の特徴抽出
    query_keypoints, query_descriptors = sift.detectAndCompute(query_img_gray, None)
    if not has_enough_keypoints(query_keypoints): return None
    # 候補画像群と比較
    for idx, candidate_img_gray in enumerate(candidate_imgs_gray):
        # 候補画像の特徴抽出
        candidate_keypoints, candidate_descriptors = sift.detectAndCompute(candidate_img_gray, None)
        if not has_enough_keypoints(candidate_keypoints): return None
        # k個の最近傍を返す
        matches = bf_matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
        ratio_inliers = [
            m for m, n in matches
            if ratio_test(m.distance, n.distance, ratio_thresh)
        ]
        ransac_inliers = (query_keypoints, candidate_keypoints, ratio_inliers, ransac_thresh)
        num_of_matches = len(ransac_inliers)
        # 類似画像の更新
        if num_of_matches > max_matches and num_of_matches >= match_thresh:
            max_matches = num_of_matches
            most_similar_img_idx = idx
            most_similar_img = candidate_img_gray
    return most_similar_img_idx, most_similar_img

def search_similar_img_by_colorhist(query_img_BGR, candidate_imgs_BGR):
    if is_grayscale(query_img_BGR):
        TypeError("クエリ画像にBGR画像を入力してください")
    if is_grayscale(candidate_imgs_BGR):
        TypeError("候補画像にBGR画像を入力してください")
    most_similar_img_idx = None    
    most_similar_img = None
    best_score = -1
    #[0,1,2]：対象チャンネル（B=0, G=1, R=2）, [8, 8, 8]:各チャンネルのビン数, [0, 256, 0, 256, 0, 256]:各チャンネルの値の範囲
    query_hist = cv2.calcHist([query_img_BGR], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    query_hist_norm = cv2.normalize(query_hist, query_hist).flatten() # スケールの影響を軽減するため正規化
    # 候補画像と比較
    for idx, candidate_img_BGR in enumerate(candidate_imgs_BGR):
        cand_hist = cv2.calcHist([candidate_img_BGR], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cand_hist_norm = cv2.normalize(cand_hist, cand_hist).flatten()
        # 類似度計算
        score = cv2.compareHist(query_hist_norm, cand_hist_norm, cv2.HISTCMP_INTERSECT)
        if score > best_score:
            best_score = score
            most_similar_img_idx = idx
            most_similar_img = candidate_img_BGR
    return most_similar_img_idx, most_similar_img

if __name__ == '__main__':
    
    import glob
    import os
    import img_utils as iu

    query_img_path = "../input/ssi_test/query_img/camera_0001_0001.jpg"
    candidate_imgs_path = "../input/ssi_test/candidate_imgs"

    query_img = cv2.imread(query_img_path)

    candidate_img_paths = glob.glob(os.path.join(candidate_imgs_path, "*.jpg"))
    candidate_imgs = [cv2.imread(p) for p in candidate_img_paths]

    most_similar_idx, most_similar_img = search_similar_img_by_colorhist(query_img, candidate_imgs)
    if most_similar_img is None:
        print("類似画像なし")
    else:
        iu.show_imgs(most_similar_img)
