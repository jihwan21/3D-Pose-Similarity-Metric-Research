# psim.py

import numpy as np

class PSIM:
    def __init__(self, anchor, joint1, joint2, joint3, position):
        self.anchor = anchor
        self.joint1 = joint1
        self.joint2 = joint2
        self.joint3 = joint3

        # if position == p1:
        #     joint1 = joints3d[0][12] # Neck
        #     joint2 = joints3d[0][16] # L_Shoulder
        #     joint3 = joints3d[0][18] # L_Elbow
        #     joint4 = joints3d[0][20] # L_Wrist
        # elif position == p2:
        #     joint1 = joints3d[0][12] # Neck
        #     joint2 = joints3d[0][16] # L_Shoulder
        #     joint3 = joints3d[0][18] # L_Elbow
        #     joint4 = joints3d[0][20] # L_Wrist
    
    # anchor joint 기준 다른 joint와의 구면 좌표값 계산 함수 + MSCN
    def anchor_spherical_mscn_cal(self, anchor, joint1, joint2, joint3):
        spherical_list = []
        joint_list = [joint1, joint2, joint3]
        
        for vec in joint_list:
            lst = []
            anchor = self.anchor.cpu().numpy()  # CUDA Tensor를 호스트 메모리로 복사한 후 NumPy 배열로 변환
            vec = vec.cpu().numpy()

            r, theta, phi = cartesian_to_spherical(anchor, vec)
            lst.append(r)
            lst.append(theta)
            lst.append(phi)

            spherical_list.append(lst)
    
    # 앞서 구한 구면 좌표값 중 극값과 방위각 MSCN 정규화
        joint1 = spherical_list[0]
        joint2 = spherical_list[1] 
        joint3 = spherical_list[2]
        
        theta = [joint1[1], joint2[1], joint3[1]]
        phi = [joint1[2], joint2[2], joint3[2]]
        
        theta_mean = np.mean(theta)
        phi_mean = np.mean(phi)
        theta_std = np.std(theta)
        phi_std = np.std(phi)
        
        mscn_coeff = mscn_norm(theta, phi, theta_mean, phi_mean, theta_std, phi_std)
        
        return mscn_coeff
    
def cartesian_to_spherical(p1, p2):
    # 벡터 p1에서 p2로 향하는 벡터 계산
    vector = p2 - p1
    
    # 극좌표에서의 반지름, azimuth 각도, elevation 각도 계산
    r = np.linalg.norm(vector)
    theta = np.arctan2(vector[1], vector[0])
    phi = np.arccos(vector[2] / r) if r != 0 else 0  # 방지: 벡터의 길이가 0인 경우

    return r, theta, phi

def mscn_norm(theta, phi, theta_mean, phi_mean, theta_std, phi_std):
    mscn_theta = [(t - theta_mean) / theta_std for t in theta]
    mscn_phi = [(t - phi_mean) / phi_std for t in phi]
    mscn_coeff = mscn_theta + mscn_phi
    return mscn_coeff

# if __name__=="__main__":
#     joint1 = [-0.02234, -0.84803, -0.053134] # Neck
#     joint2 = [0.16416, -0.788, -0.10793] # L_Shoulder
#     joint3 = [0.16247, -0.52108, -0.082315] # L_Elbow
#     joint4 = [-0.01515, -0.41851, -0.23507] # L_Wrist

#     psim = PSIM(joint1, joint2, joint3, joint4)
#     neck_mscn_coeff = psim.anchor_spherical_mscn_cal(joint1, joint2, joint3, joint4)
#     print(neck_mscn_coeff)

    
