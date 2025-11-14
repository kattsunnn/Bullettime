import math
import numpy as np

class PPI:
    def __init__(self, img_e, img_p, angle_u, angle_v):
        self.img_e = img_e
        self.img_e_h = img_e.shape[0]
        self.img_e_w = img_e.shape[1]

        self.img_p = img_p
        self.img_p_h = img_p.shape[0]
        self.img_p_w = img_p.shape[1]

        self.angle_u = angle_u
        self.angle_v = angle_v

    def get_ppi(self):
        return self.img_p
    def get_src_img(self):
        return self.img_e
    def get_ppi_h(self):
        return self.img_p_h
    def get_ppi_w(self):
        return self.img_p_w
    def get_angle_u(self):
        return self.angle_u
    def get_angle_v(self):
        return self.angle_v
    def get_focal_length(self):
        return self.img_e_w / (2*np.pi)

    # 透視投影画像の座標から全方位画像の角度座標を得る
    def get_angular_coordinate(self, u_p, v_p):
        # 透視投影画像の画像座標から回転前の3次元視線ベクトルを計算
        focal_length = self.img_e_w/(2*np.pi)
        x = u_p-(self.img_p_w/2)
        y = v_p-(self.img_p_h/2)
        z = focal_length
        # 透視投影画像の生成に使用した回転行列を計算
        angle_u_rad = np.deg2rad(self.angle_u)
        angle_v_rad = np.deg2rad(self.angle_v)
        R = np.dot(PPI.rotation_y(angle_u_rad), PPI.rotation_x(angle_v_rad))
        # 回転後の視線ベクトルを計算
        rotated_x = R[0][0] * x + R[0][1] * y + R[0][2] * z
        rotated_y = R[1][0] * x + R[1][1] * y + R[1][2] * z
        rotated_z = R[2][0] * x + R[2][1] * y + R[2][2] * z
        # 回転した視線ベクトルから全方位画像の角度座標を計算
        theta_e = np.arctan2(rotated_x, rotated_z)
        phi_e = np.arctan2(np.sqrt(rotated_x**2 + rotated_z**2), rotated_y)

        return np.rad2deg(theta_e), np.rad2deg(phi_e)
    
    # X軸周りの回転行列
    @staticmethod
    def rotation_x(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[1.0, 0.0, 0.0],
                    [0.0, cos_a, -sin_a],
                    [0.0, sin_a, cos_a]], dtype=np.float64)
        return R
    # Y軸周りの回転行列
    @staticmethod
    def rotation_y(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[cos_a, 0.0, sin_a],
                    [0.0, 1.0, 0.0],
                    [-sin_a, 0.0, cos_a]], dtype=np.float64)
        return R
    

    

