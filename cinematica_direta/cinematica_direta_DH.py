"""
Validação da Cinemática Direta do UR5 - DH Calibrada
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERRO] Instale: pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)

plt.style.use("seaborn-v0_8-whitegrid")

PASTA_SAIDA = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(PASTA_SAIDA, exist_ok=True)

NOMES_JUNTAS = [f"/UR5_joint{i}" for i in range(1, 7)]
NOME_EFETUADOR = "/UR5_connection"
NOME_BASE = "/UR5"

# =============================================================================
# TABELA DH CALIBRADA com base no seu modelo do CoppeliaSim
# =============================================================================
DH_PARAMS = [
    [0.00000,  0.089159,  np.pi/2],    # J1
    [-0.42500, 0.00000,   0.0     ],   # J2  (ajustado)
    [-0.39225, 0.00000,   0.0     ],   # J3
    [0.00000,  0.10915,   np.pi/2 ],   # J4
    [0.00000,  0.09465,  -np.pi/2 ],   # J5
    [0.00000,  0.08230,   0.0     ]    # J6
]

CONFIGURACOES_TESTE_DEG = [  # mesmo do original
    [10, -70, 80, -90, 100, 20],
    [-25, -55, 70, -105, 95, -15],
    [40, -80, 100, -60, 80, 35],
    [-50, -65, 85, -110, 110, -40],
    [70, -50, 60, -100, 75, 50],
    [-15, -95, 110, -55, 120, -25],
    [55, -40, 50, -120, 60, 65],
    [-65, -75, 95, -70, 130, -55],
    [25, -110, 130, -45, 70, 10],
    [-35, -85, 95, -85, 105, -30],
]

# =============================================================================
# FUNÇÕES
# =============================================================================

def matriz_dh(theta, d, a, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])


def calcula_pose_dh(thetas):
    T = np.eye(4)
    for i in range(6):
        a, d, alpha = DH_PARAMS[i]
        Ai = matriz_dh(thetas[i], d, a, alpha)
        T = T @ Ai
    return T


def quaternion_para_rotacao(q):
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ])


def rotacao_para_euler_xyz(R):
    pitch = np.arcsin(np.clip(-R[2,0], -1, 1))
    if np.isclose(np.cos(pitch), 0, atol=1e-8):
        roll = 0
        yaw = np.arctan2(-R[0,1], R[1,1])
    else:
        roll = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(R[1,0], R[0,0])
    return np.degrees([roll, pitch, yaw])


class InterfaceRoboCoppelia:
    def __init__(self, host="localhost", port=23000):
        print(f"[INFO] Conectando ao CoppeliaSim...")
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require("sim")
        print("[INFO] Conexão OK!")
        self.handles_juntas = [self.sim.getObject(n) for n in NOMES_JUNTAS]
        self.handle_efetuador = self.sim.getObject(NOME_EFETUADOR)
        self.handle_base = self.sim.getObject(NOME_BASE)

    def mover_juntas(self, thetas_rad):
        for h, theta in zip(self.handles_juntas, thetas_rad):
            self.sim.setJointPosition(h, float(theta))
        time.sleep(0.15)

    def ler_pose_efetuador(self):
        pos = np.array(self.sim.getObjectPosition(self.handle_efetuador, self.handle_base))
        q = self.sim.getObjectQuaternion(self.handle_efetuador, self.handle_base)
        R = quaternion_para_rotacao(q)
        return pos, R

    def coletar_ground_truth(self):
        n = len(CONFIGURACOES_TESTE_DEG)
        thetas_todos = np.radians(CONFIGURACOES_TESTE_DEG)
        pos_gt = np.zeros((n, 3))
        rot_gt = np.zeros((n, 3, 3))

        print(f"\n[INFO] Coletando Ground Truth — {n} configs...")
        for i, thetas in enumerate(thetas_todos):
            self.mover_juntas(thetas)
            pos, R = self.ler_pose_efetuador()
            pos_gt[i] = pos
            rot_gt[i] = R
            print(f"  [{i+1:2d}/{n}] θ={np.degrees(thetas).round(1)}° → pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        return {"thetas": thetas_todos, "pos_gt": pos_gt, "rot_gt": rot_gt}


def avaliar_precisao_direta(gt):
    n = len(gt["thetas"])
    erro_pos = np.zeros(n)
    erro_ori = np.zeros(n)
    pos_calc = np.zeros((n, 3))

    print("[INFO] Calculando Cinemática Direta com DH...")
    for i in range(n):
        T = calcula_pose_dh(gt["thetas"][i])
        p = T[:3, 3]
        R = T[:3, :3]
        pos_calc[i] = p
        erro_pos[i] = np.linalg.norm(p - gt["pos_gt"][i])
        erro_ori[i] = np.linalg.norm(R - gt["rot_gt"][i], ord='fro')

    return erro_pos, erro_ori, pos_calc


def imprimir_relatorio(gt, erro_pos, erro_ori):
    print("\n" + "="*75)
    print("   RELATÓRIO - CINEMÁTICA DIRETA (Tabela DH)")
    print("="*75)
    print(f"Erro médio posição : {np.mean(erro_pos)*1000:.4f} mm")
    print(f"Erro máximo posição: {np.max(erro_pos)*1000:.4f} mm")
    print(f"Erro médio orientação: {np.mean(erro_ori):.4f}")
    print("="*75)


if __name__ == "__main__":
    try:
        robo = InterfaceRoboCoppelia()
    except Exception as e:
        print(f"[ERRO] {e}")
        sys.exit(1)

    gt = robo.coletar_ground_truth()
    erro_pos, erro_ori, pos_calc = avaliar_precisao_direta(gt)
    imprimir_relatorio(gt, erro_pos, erro_ori)

    print(f"\n Pronto! Erros calculados.")