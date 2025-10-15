import math
import numpy as np
import cv2
import pdb
#
import e2p
import pd

class PPI:
    def __init__(self, img_e, img_p, angle_u, angle_v):
        self.img_e = img_e
        self.img_e_h = img_e.shape[0]
        self.img_e_w = img_e.shape[1]

        self.img_p = img_p
        self.img_p_h = img_p.shape[0]
        self.img_p_w = img_p.shape[1]

        self.angle_u = DEG2RAD(angle_u)
        self.angle_v = DEG2RAD(angle_v)

    def get_ppi(self):
        return self.img_p
    
    def get_ppi_h(self):
        return self.img_p_h

    def get_ppi_w(self):
        return self.img_p_w
    
    def get_angular_coordinate(self, u_p, v_p):
        
        # 画像座標から３次元視線ベクトルへの変換
        focal_length = self.img_e_w/(2*np.pi)
        x = u_p-(self.img_p_w/2)
        y = v_p-(self.img_p_h/2)
        z = focal_length
        # 正規化
        # vec = np.array([x, y, z])
        # norm = np.linalg.norm(vec)
        # if norm != 0:
        #     vec_normalized = vec / norm
        # else:
        #     vec_normalized = vec  # ゼロベクトルの場合

        # 回転行列で回転
        R = np.dot(self.rotation_y(self.angle_u), self.rotation_x(self.angle_v))
        # rotated_vec = np.dot(R, vec_normalized)

        # Xx, Xy, Xz = rotated_vec

        Xx = R[0][0] * x + R[0][1] * y + R[0][2] * z
        Xy = R[1][0] * x + R[1][1] * y + R[1][2] * z
        Xz = R[2][0] * x + R[2][1] * y + R[2][2] * z

        # 視線ベクトルから角度座標を計算
        theta_e = np.arctan2(Xx, Xz)
        phi_e = np.arctan2(np.sqrt(Xx**2 + Xz**2), Xy)

        return RAD2DEG(theta_e), RAD2DEG(phi_e)
    
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

def RAD2DEG(a):
    return a * 180.0 / np.pi
# 角度の単位変換
def DEG2RAD(a):
    return a * np.pi / 180.0