import cv2

def bilinear_interpolation(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = h * scale, w * scale
    # 双一次補間
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return scaled_img

def fsrcnn(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("models/FSRCNN_x4.pb")  # 学習済みモデル
    sr.setModel("fsrcnn", 4)      # モデル名と倍率
    scaled_img = sr.upsample(img)     # 画像に適用
    return scaled_img

if __name__ == '__main__':
    # 結果を表示
    import img_utils as iu

    # scale = sys.argv[1]
    input_path, output_path = iu.prepare_io_paths()    
    img  = iu.load_imgs(input_path)
    scaled_img = fsrcnn(img)
    iu.save_imgs(scaled_img, output_path)
    # iu.show_imgs(super_resolution(img))



