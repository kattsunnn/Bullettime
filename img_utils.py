import os
import cv2
import numpy as np
import argparse
import glob

def prepare_io_paths(description="入出力の共通引数を処理"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", required=True, help="入力ファイルまたはフォルダのパス")
    parser.add_argument("-o", "--output", required=True, default="outputs", help="出力ディレクトリパス")
    args = parser.parse_args()

    # 絶対パスに変換
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    # 出力フォルダ作成
    os.makedirs(output_path, exist_ok=True)

    return input_path, output_path

def load_imgs(path):

    if os.path.isfile(path):
        img = cv2.imread(path)
        return img

    elif os.path.isdir(path):
        img_paths = []
        exts = ["jpg", "jpeg", "png"]
        for ext in exts:
            img_paths += glob.glob(os.path.join(path, f"*.{ext}"))
            img_paths += glob.glob(os.path.join(path, f"*.{ext.upper()}"))
        imgs = [cv2.imread(img_path) for img_path in img_paths ]
        return imgs
    
    else:
        raise ValueError(f"入力パスが存在しません: {path}")

def save_imgs(imgs, output_path, file_name_pattern=f"img_{{}}", expand=".jpg"):
    os.makedirs(output_path, exist_ok=True)
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    for i, img in enumerate(imgs):
        file_name = file_name_pattern.format(f"{i:04d}") + expand
        file_path = os.path.join(output_path, file_name)
        cv2.imwrite(file_path, img)
        print(f"{file_path} を保存しました。")

def show_imgs(imgs):
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    for idx, img in enumerate(imgs):
        # ウィンドウ名を画像ごとに付ける
        win_name = f"img_{idx}" if len(imgs) > 1 else "img"
        cv2.imshow(win_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




    