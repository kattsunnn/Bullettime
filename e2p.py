# Todo: 変数名をリファクタリング
# Todo: リアルタイム処理を想定しているため、画像出力の処理が冗長？バレットタイムように調整

import math
import numpy as np
import cv2

class E2P:
    def __init__(self, _src_size_w, _src_size_h, _dst_w, _dst_h):

        # 全方位画像の大きさ
        self.Sw = _src_size_w
        self.Sh = _src_size_h

        # 透視投影画像の大きさ
        self.ow = _dst_w
        self.oh = _dst_h
        self.owh = 0.5 * self.ow
        self.ohh = 0.5 * self.oh

        # 焦点距離
        self.f = self.Sw / (2*np.pi)
        
        # remap用変数
        self.map_u = np.zeros((_dst_h, _dst_w), dtype=np.float32)
        self.map_v = np.zeros((_dst_h, _dst_w), dtype=np.float32)

        # 内挿関数指定
        self.interp_method = cv2.INTER_LINEAR

    # X軸周りの回転行列
    def rotation_x(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[1.0, 0.0, 0.0],
                    [0.0, cos_a, -sin_a],
                      [0.0, sin_a, cos_a]], dtype=np.float64)
        return R

    # Y軸周りの回転行列
    def rotation_y(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[cos_a, 0.0, sin_a],
                      [0.0, 1.0, 0.0],
                      [-sin_a, 0.0, cos_a]], dtype=np.float64)
        return R

    # 任意の軸周りの回転行列
    def rotation_by_axis(self, n, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        l_cos_a = 1.0 - cos_a
        R = np.array([[cos_a + n[0] * n[0] * l_cos_a,
                       n[0] * n[1] * l_cos_a - n[2] * sin_a,
                       n[2] * n[0] * l_cos_a + n[1] * sin_a],
                      [n[0] * n[1] * l_cos_a + n[2] * sin_a,
                       cos_a + n[1] * n[1] * l_cos_a,
                       n[1] * n[2] * l_cos_a - n[0] * sin_a],
                      [n[2] * n[0] * l_cos_a - n[1] * sin_a,
                       n[1] * n[2] * l_cos_a + n[0] * sin_a,
                       cos_a + n[2] * n[2] * l_cos_a]], dtype=np.float64)
        return R

    #
    # 画像生成用のマップ作成関数
    #
    def generate_map(self,
                     angle_u,
                     angle_v,
                     angle_z,
                     scale=1):
        # 回転行列の計算
        R = np.dot(self.rotation_y(angle_u), self.rotation_x(angle_v))
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        Ri = self.rotation_by_axis(np.dot(R, axis), angle_z)
        R = np.dot(Ri, R)
        
        # 一括計算用の出力画像の画素座標データ
        u = np.arange(0, self.ow, 1)
        v = np.arange(0, self.oh, 1)
        dst_u, dst_v = np.meshgrid(u, v)

        # 視線ベクトルを計算
        x = dst_u - self.owh
        y = dst_v - self.ohh 
        z = self.f * scale * np.ones((self.oh, self.ow))
        
        # 回転行列で回転
        Xx = R[0][0] * x + R[0][1] * y + R[0][2] * z
        Xy = R[1][0] * x + R[1][1] * y + R[1][2] * z
        Xz = R[2][0] * x + R[2][1] * y + R[2][2] * z
        
        # 視線ベクトルから角度を計算
        theta = np.arctan2(Xx, Xz)
        phi = np.arctan2(np.sqrt(Xx**2 + Xz**2), Xy)
        
        # # 角度から入力画像の座標を計算
        # self.map_u = ((theta+np.pi)*(self.Sw/(2*np.pi))).astype(np.float32)
        # self.map_v = (-(phi-np.pi)*(self.Sh/np.pi)).astype(np.float32)

        # 角度から入力画像の座標を計算
        self.map_u = (0.5 * (theta + np.pi) * self.Sw / np.pi - 0.5).astype(np.float32)
        self.map_v = ((np.pi - phi) * self.Sh / np.pi - 0.5).astype(np.float32)
    #
    # 画像生成
    #
    def generate_image(self, src_img):
        return cv2.remap(src_img, self.map_u, self.map_v, self.interp_method)
