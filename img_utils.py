import os
import cv2
import numpy as np

def save_imgs(imgs, output_path, file_name_pattern=f"img_{{}}", expand=".jpg"):
    os.makedirs(output_path, exist_ok=True)
    for i, img in enumerate(imgs):
        file_name = file_name_pattern.format(f"{i+1:04d}") + expand
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