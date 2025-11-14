import numpy as np
import os
import argparse
import cv2
import e2p
from pd import PD
from ppi import PPI

# 透視投影画像群の生成
def generate_ppis(img_e):
    # 使用する全方位カメラの解像度
    src_size_h = img_e.shape[0]
    src_size_w = img_e.shape[1]
    fov = 20  # 透視投影画像の視野角
    # 表示する透視投影画像の解像度
    dst_width = int(2.0 * np.tan(np.deg2rad(fov) / 2.0) * src_size_w / (2*np.pi))
    dst_height = int(2.0 * np.tan(np.deg2rad(fov) / 2.0) * src_size_h / np.pi)
    # 視線角度の設定
    # theta_eyes = [-60, -30, 0, 30, 60]
    # theta_eyes = [120, 150, 180, 210, 240]
    # theta_eyes = [150, 157.5, 165, 172.5, 180, 187.5, 195, 202.5, 210] # 15度半重複 7.5度刻み
    theta_eyes = [0] 
    phi_eyes = [0]

    ppis = []

    for j, phi_eye in enumerate(phi_eyes):
        for i, theta_eye in enumerate(theta_eyes):
            view = e2p.E2P(src_size_w, src_size_h, dst_width, dst_height)
            view.generate_map(np.deg2rad(theta_eye), np.deg2rad(phi_eye), 0, 1)
            dst_img = view.generate_image(img_e)
            ppi = PPI(img_e, dst_img, theta_eye, phi_eye)
            ppis.append(ppi)
    return ppis

# メイン関数
def main():
    parser = argparse.ArgumentParser(description="透視投影画像の解像度を確認")
    parser.add_argument("-i", "--input", required=True, help="入力動画ファイルパス")
    parser.add_argument("-o", "--output", required=False, default="outputs/check_resolution", help="出力ディレクトリパス")
    args = parser.parse_args()

    # 入力画像のパスを読み込み
    img_e_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 入力画像の読み込み
    img_e = cv2.imread(img_e_path)
    # 透視投影画像群の生成
    ppis = generate_ppis(img_e)

    for ppi_num, ppi in enumerate(ppis):
        pose_detector_for_ppi = PD(ppi.get_ppi())
        if pose_detector_for_ppi.is_pose_detected():
            drawn_landmarks = pose_detector_for_ppi.draw_pose_landmarks()
            save_path = os.path.join(output_dir, f"ppi_{ppi_num}_landmarks.png")
            cv2.imwrite(save_path, drawn_landmarks)
        save_path = os.path.join(output_dir, f"ppi_{ppi_num}.png")
        cv2.imwrite(save_path, ppi.get_ppi())
    print("finish")
if __name__ == '__main__':
    main()



