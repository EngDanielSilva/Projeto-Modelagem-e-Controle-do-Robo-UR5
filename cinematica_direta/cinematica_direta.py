"""
VALIDAÇÃO DA CINEMÁTICA DIRETA DO UR5
=====================================
Objetivo: Validar se o cálculo matemático da Cinemática Direta em Python
          corresponde à pose real obtida no simulador CoppeliaSim.

Este script:
  1. Conecta ao CoppeliaSim e move o robô para várias configurações.
  2. Coleta a pose real (Ground Truth) do efetuador no simulador.
  3. Calcula a mesma pose usando matemática em Python.
  4. Compara os resultados (erros de posição e orientação).
  5. Gera gráficos e relatório para análise.
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
# IMPORTAÇÕES
# ===================================================================
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERRO] O pacote 'coppeliasim-zmqremoteapi-client' não está instalado.")
    print("       Instale com: pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)

# Configuração do estilo dos gráficos (mais bonito e profissional)
plt.style.use("seaborn-v0_8-whitegrid")

# ===================================================================
# CONFIGURAÇÕES GERAIS
# ===================================================================

# Pasta onde serão salvos os gráficos e arquivos gerados
PASTA_SAIDA = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Nomes dos objetos na cena do CoppeliaSim (devem ser iguais aos da cena)
NOMES_JUNTAS = [f"/UR5_joint{i}" for i in range(1, 7)]   # Juntas 1 até 6
NOME_EFETUADOR = "/UR5_connection"                       # Garra / End-effector
NOME_BASE = "/UR5"                                       # Base do robô


# ===================================================================
# CADEIA CINEMÁTICA REAL EXTRAÍDA DO SIMULADOR
# ===================================================================
"""
Esta é a parte mais importante deste código.

Em vez de usar a tabela DH tradicional (método teórico), 
nós extraímos diretamente do CoppeliaSim a geometria real dos elos 
quando todas as juntas estão em ZERO graus.

Cada item da lista contém:
    - pos  = posição (x, y, z) do frame do elo
    - quat = orientação (quaternion) do frame do elo

Vantagem: Muito mais preciso para este modelo específico do simulador.
"""
CADEIA_ZERO = [
    # Elo 1 (Base)
    ([0.0, 0.0, 0.0085], [0.0, 0.0, 0.0, 1.0]),
    
    # Elo 2
    ([-0.070317, -1.4e-05, 0.066039], [6.8e-05, -0.707159, 6.8e-05, 0.707054]),
    
    # Elo 3
    ([0.425105, -9.1e-05, 1.9e-05], [-0.0, -1.5e-05, -0.000107, 1.0]),
    
    # Elo 4
    ([0.392149, -7.6e-05, 2e-06], [-0.0, -0.0, -9.7e-05, 1.0]),
    
    # Elo 5
    ([0.045573, -1e-05, 0.03971], [7.5e-05, 0.707098, -7.5e-05, 0.707116]),
    
    # Elo 6
    ([0.014424, -4e-06, 0.049176], [-8.5e-05, -0.707107, -8.5e-05, 0.707107]),
]

# Transformação final da ferramenta/garra
EFETUADOR_ZERO = ([-0.0, -0.0, 0.089375], [-0.0, -0.0, -9.7e-05, 1.0])


# ===================================================================
# CONFIGURAÇÕES DE TESTE
# ===================================================================
"""
Conjunto de 10 configurações bem escolhidas para teste.

Critérios de escolha (importante para o relatório):
- Evitar singularidades (junta 3 e junta 5 longe de 0°/180°)
- Manter o robô dentro do espaço de trabalho útil
- Evitar colisões com a mesa ou com o próprio robô
- Boa distribuição no espaço
"""
CONFIGURACOES_TESTE_DEG = [
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
# FUNÇÕES MATEMÁTICAS AUXILIARES
# =============================================================================
"""
Nesta seção estão todas as funções matemáticas necessárias para 
realizar a Cinemática Direta.

Importante: Estas funções são independentes do simulador e representam 
a teoria por trás do cálculo.
"""

def quaternion_para_rotacao(q):
    """
    Converte um quaternion (qx, qy, qz, qw) em matriz de rotação 3x3.
    
    O quaternion é uma forma eficiente de representar orientações no espaço 3D,
    evitando problemas de gimbal lock (comum em ângulos de Euler).
    """
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


def rotacao_em_z(theta):
    """
    Cria uma matriz homogênea 4x4 de rotação pura em torno do eixo Z.
    
    Como todas as juntas do UR5 são revolutivas em torno do eixo Z local,
    usamos esta função para aplicar o ângulo θi de cada junta.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    T = np.eye(4)                    # Matriz identidade 4x4
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def transformacao_homogenea(pos, quat):
    """
    Monta a matriz de transformação homogênea 4x4 completa 
    a partir de uma posição (x,y,z) e uma orientação (quaternion).
    
    Esta matriz combina rotação + translação em um único objeto matemático.
    """
    T = np.eye(4)
    T[:3, :3] = quaternion_para_rotacao(quat)   # Parte de rotação (3x3)
    T[:3, 3] = pos                               # Parte de translação (posição)
    return T


def rotacao_para_vetor(R):
    """
    Converte uma matriz de rotação em vetor ângulo-eixo (axis-angle).
    Útil para calcular erro de orientação de forma mais intuitiva.
    
    (Esta função não é usada nos gráficos principais, mas está disponível.)
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-10:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        diag = np.array([R[0, 0] + 1, R[1, 1] + 1, R[2, 2] + 1]) / 2
        eixo = np.sqrt(np.maximum(diag, 0))
        eixo[0] *= np.sign(R[0, 1] + R[1, 0] + 1e-10)
        eixo[1] *= np.sign(R[0, 2] + R[2, 0] + 1e-10)
        eixo /= (np.linalg.norm(eixo) + 1e-10)
        return theta * eixo
    return theta / (2 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])


def rotacao_para_euler_xyz(R):
    """
    Converte matriz de rotação para ângulos de Euler (Roll, Pitch, Yaw) em graus.
    
    ⚠️ Atenção: Usada APENAS para visualização nos gráficos de orientação.
    Não é usada no cálculo principal da cinemática (evita problemas de singularidade).
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.isclose(np.cos(pitch), 0.0, atol=1e-8):
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


def calcula_pose_efetuador(thetas):
    """
    CINEMÁTICA DIRETA - Função Principal
    
    Calcula a pose (posição + orientação) do efetuador a partir dos ângulos 
    das 6 juntas (thetas em radianos).
    
    Como funciona (multiplicação sequencial de matrizes):
        T = Identidade
        Para cada elo i de 0 até 5:
            T = T × T_fixo_do_elo_i × Rotação_Z(θi)
        T = T × T_efetuador_final
    """
    T = np.eye(4)   # Começa na base do robô (frame 0)

    # Percorre os 6 elos da cadeia cinemática
    for i, (pos, quat) in enumerate(CADEIA_ZERO):
        T_fixo = transformacao_homogenea(pos, quat)      # Transformação fixa do elo (extraída do simulador)
        T_junta = rotacao_em_z(thetas[i])                # Rotação variável da junta i
        T = T @ T_fixo @ T_junta                         # Composição das transformações

    # Aplica a transformação da ferramenta/garra no final
    T = T @ transformacao_homogenea(*EFETUADOR_ZERO)
    
    return T


# =============================================================================
# INTERFACE COM O COPPELIASIM
# =============================================================================
"""
Esta classe é responsável por toda a comunicação entre o Python e o simulador 
CoppeliaSim. Ela faz o "meio de campo" prático da validação.
"""

class InterfaceRoboCoppelia:
    """
    Classe que gerencia a conexão, movimentação e leitura de dados do robô 
    no simulador CoppeliaSim.
    """

    def __init__(self, host="localhost", port=23000):
        """
        Inicializa a conexão com o CoppeliaSim.
        """
        print(f"[INFO] Conectando ao CoppeliaSim em {host}:{port}...")
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require("sim")          # Objeto principal do simulador
        print("[INFO] Conexão estabelecida com sucesso!")
        self._resolver_handles()

    def _resolver_handles(self):
        """
        Obtém os 'handles' (identificadores) de todos os objetos importantes 
        na cena do CoppeliaSim. 
        Necessário para controlar as juntas e ler a pose do efetuador.
        """
        self.handles_juntas = [self.sim.getObject(n) for n in NOMES_JUNTAS]
        self.handle_efetuador = self.sim.getObject(NOME_EFETUADOR)
        self.handle_base = self.sim.getObject(NOME_BASE)
        print("[INFO] Handles dos objetos resolvidos.")

    def mover_juntas(self, thetas_rad):
        """
        Move todas as 6 juntas do robô para os ângulos especificados (em radianos).
        
        Parâmetro:
            thetas_rad: array com 6 ângulos em radianos
        """
        for h, theta in zip(self.handles_juntas, thetas_rad):
            self.sim.setJointPosition(h, float(theta))
        
        time.sleep(0.05)   # Pequena pausa para o simulador atualizar fisicamente

    def ler_pose_efetuador(self):
        """
        Lê a pose atual (posição + orientação) do efetuador no simulador.
        Esta é a nossa 'Ground Truth' (verdade real).
        
        Retorna:
            pos: posição [x, y, z]
            R:   matriz de rotação 3x3
            T:   matriz homogênea 4x4 completa
        """
        # Lê posição e orientação em relação à base do robô
        pos = np.array(self.sim.getObjectPosition(self.handle_efetuador, self.handle_base))
        q = self.sim.getObjectQuaternion(self.handle_efetuador, self.handle_base)
        
        R = quaternion_para_rotacao(q)          # Converte quaternion para matriz de rotação
        
        # Monta matriz homogênea completa
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        
        return pos, R, T

    def coletar_ground_truth(self, configs_deg=None):
        """
        Função principal de coleta de dados:
        - Move o robô para cada configuração de teste
        - Lê a pose real (Ground Truth)
        - Armazena tudo para posterior comparação
        """
        if configs_deg is None:
            configs_deg = CONFIGURACOES_TESTE_DEG
            
        n = len(configs_deg)
        thetas_todos = np.radians(configs_deg)   # Converte graus para radianos
        
        pos_gt = np.zeros((n, 3))
        rot_gt = np.zeros((n, 3, 3))
        T_gt = []

        print(f"\n[INFO] Coletando Ground Truth — {n} configurações...")
        
        for i, thetas in enumerate(thetas_todos):
            self.mover_juntas(thetas)                    # Move o robô
            pos, R, T = self.ler_pose_efetuador()        # Lê a pose real
            
            pos_gt[i] = pos
            rot_gt[i] = R
            T_gt.append(T)
            
            print(f"  [{i + 1:2d}/{n}] θ={np.degrees(thetas).round(1)}°  "
                  f"→ pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m")
        
        print("[INFO] Coleta de Ground Truth concluída!\n")
        return {
            "thetas": thetas_todos,
            "pos_gt": pos_gt,
            "rot_gt": rot_gt,
            "T_gt": T_gt
        }

    def salvar_csv(self, gt, caminho):
        """Salva os dados do Ground Truth em formato CSV (útil para análise posterior)."""
        cabecalho = ("theta1,theta2,theta3,theta4,theta5,theta6,"
                     "px,py,pz,R00,R01,R02,R10,R11,R12,R20,R21,R22")
        
        linhas = [np.concatenate([gt["thetas"][i], 
                                  gt["pos_gt"][i], 
                                  gt["rot_gt"][i].flatten()])
                  for i in range(len(gt["thetas"]))]
        
        np.savetxt(caminho, linhas, delimiter=",", header=cabecalho, 
                   comments="", fmt="%.8f")
        print(f"[INFO] CSV salvo em: {caminho}")


# =============================================================================
# ANÁLISE DE ERROS (Python vs Ground Truth)
# =============================================================================

# =============================================================================
# ANÁLISE DE ERROS (Versão Melhorada)
# =============================================================================

def avaliar_precisao_direta(gt):
    """
    Calcula os erros entre a Cinemática Direta em Python e o Ground Truth.
    
    Retorna:
        erro_pos     : erro total de posição (norma euclidiana)
        erro_pos_x/y/z : erro individual por eixo
        erro_ori     : erro de orientação
        pos_calc, rot_calc : valores calculados
    """
    n = len(gt["thetas"])
    
    # Erros de posição
    erro_pos = np.zeros(n)      # Erro total (distância 3D)
    erro_pos_x = np.zeros(n)
    erro_pos_y = np.zeros(n)
    erro_pos_z = np.zeros(n)
    
    erro_ori = np.zeros(n)
    
    pos_calc = np.zeros((n, 3))
    rot_calc = np.zeros((n, 3, 3))

    print("[INFO] Calculando Cinemática Direta em Python e erros...")

    for i in range(n):
        T = calcula_pose_efetuador(gt["thetas"][i])
        p = T[:3, 3]
        R = T[:3, :3]
        
        pos_calc[i] = p
        rot_calc[i] = R
        
        diff = p - gt["pos_gt"][i]                    # Diferença em X, Y, Z
        
        erro_pos[i] = np.linalg.norm(diff)            # Erro total
        erro_pos_x[i] = abs(diff[0])
        erro_pos_y[i] = abs(diff[1])
        erro_pos_z[i] = abs(diff[2])
        
        erro_ori[i] = np.linalg.norm(R - gt["rot_gt"][i], ord='fro')

    return erro_pos, erro_pos_x, erro_pos_y, erro_pos_z, erro_ori, pos_calc, rot_calc
 

# =============================================================================
# GRÁFICOS E VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================
"""
Nesta seção estão todas as funções responsáveis por gerar os gráficos 
para análise visual da validação da Cinemática Direta.
"""
# =============================================================================
# GRÁFICOS — VERSÃO REFINADA E PROFISSIONAL
# =============================================================================

def grafico_curvas_juntas(gt, caminho):
    """Gráfico 1: Distribuição dos ângulos das juntas nos testes."""
    n = len(gt["thetas"])
    idx = np.arange(1, n + 1)
    juntas_deg = np.degrees(gt["thetas"])

    fig, ax = plt.subplots(figsize=(10, 6))
    marcadores = ["o", "s", "^", "D", "v", "P"]
    
    for j in range(6):
        ax.plot(idx, juntas_deg[:, j], 
                marker=marcadores[j], 
                label=f"Junta {j+1}", 
                linewidth=2.2, 
                markersize=7)
    
    ax.set_title("Configurações de Teste — Ângulos das Juntas", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Índice da Configuração de Teste", fontsize=12)
    ax.set_ylabel("Ângulo da Junta (graus)", fontsize=12)
    ax.set_xticks(idx)
    ax.legend(ncol=3, fontsize=10, loc='best')
    ax.grid(True, linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 1 (Juntas) salvo → {caminho}")


def grafico_curvas_posicao(gt, pos_calc, caminho):
    """Gráfico 2: Comparação de Posição (o mais importante)."""
    n = len(gt["thetas"])
    idx = np.arange(1, n + 1)
    
    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    eixos = ["X", "Y", "Z"]
    cores = ["tab:blue", "tab:green", "tab:red"]
    
    for k in range(3):
        axs[k].plot(idx, gt["pos_gt"][:, k], "o-", 
                    color=cores[k], linewidth=2.2, markersize=6,
                    label="Ground Truth (CoppeliaSim)")
        axs[k].plot(idx, pos_calc[:, k], "x--", 
                    color="black", linewidth=2.2, markersize=7,
                    label="Cálculo Python (Cinemática Direta)")
        axs[k].set_ylabel(f"Posição {eixos[k]} (m)", fontsize=12)
        axs[k].legend(fontsize=11)
        axs[k].grid(True, linestyle=":", alpha=0.8)
    
    axs[-1].set_xlabel("Índice da Configuração de Teste", fontsize=12)
    axs[-1].set_xticks(idx)
    
    fig.suptitle("Comparação de Posição do Efetuador\nGround Truth × Cálculo em Python", 
                 fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 2 (Posição) salvo → {caminho}")


def grafico_curvas_orientacao(gt, rot_calc, caminho):
    """Gráfico 3: Comparação de Orientação (Euler)."""
    n = len(gt["thetas"])
    idx = np.arange(1, n + 1)
    
    euler_gt = np.array([rotacao_para_euler_xyz(gt["rot_gt"][i]) for i in range(n)])
    euler_calc = np.array([rotacao_para_euler_xyz(rot_calc[i]) for i in range(n)])

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    nomes = ["Roll", "Pitch", "Yaw"]
    cores = ["tab:purple", "tab:orange", "tab:cyan"]
    
    for k in range(3):
        axs[k].plot(idx, euler_gt[:, k], "o-", 
                    color=cores[k], linewidth=2.2, markersize=6,
                    label="Ground Truth (CoppeliaSim)")
        axs[k].plot(idx, euler_calc[:, k], "x--", 
                    color="black", linewidth=2.2, markersize=7,
                    label="Cálculo Python")
        axs[k].set_ylabel(f"{nomes[k]} (°)", fontsize=12)
        axs[k].legend(fontsize=11)
        axs[k].grid(True, linestyle=":", alpha=0.8)
    
    axs[-1].set_xlabel("Índice da Configuração de Teste", fontsize=12)
    axs[-1].set_xticks(idx)
    
    fig.suptitle("Comparação de Orientação do Efetuador\n(Ground Truth × Cálculo em Python)", 
                 fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 3 (Orientação) salvo → {caminho}")


def grafico_erros(erro_pos, erro_ori, caminho):
    """Gráfico 4: Erros de Posição e Orientação (principal para validação)."""
    n = len(erro_pos)
    idx = np.arange(1, n + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    # Erro de Posição
    axs[0].plot(idx, erro_pos * 1000, "o-", color="firebrick", linewidth=2, markersize=7)
    axs[0].axhline(np.mean(erro_pos) * 1000, color="gray", linestyle="--", label="Média")
    axs[0].set_title("Erro de Posição", fontsize=13, fontweight='bold')
    axs[0].set_xlabel("Configuração de Teste")
    axs[0].set_ylabel("Erro (mm)")
    axs[0].set_xticks(idx)
    axs[0].legend(fontsize=11)
    axs[0].grid(True, linestyle=":")

    # Erro de Orientação
    axs[1].plot(idx, erro_ori, "s-", color="darkslateblue", linewidth=2, markersize=7)
    axs[1].axhline(np.mean(erro_ori), color="gray", linestyle="--", label="Média")
    axs[1].set_title("Erro de Orientação", fontsize=13, fontweight='bold')
    axs[1].set_xlabel("Configuração de Teste")
    axs[1].set_ylabel("Erro (Norma Frobenius)")
    axs[1].set_xticks(idx)
    axs[1].legend(fontsize=11)
    axs[1].grid(True, linestyle=":")

    fig.suptitle("Validação Numérica — Erros entre Cálculo e Ground Truth", 
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 4 (Erros) salvo → {caminho}")


def grafico_dispersao_3d(gt, pos_calc, caminho):
    """Gráfico 5: Visualização 3D das posições."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    pg, pc = gt["pos_gt"], pos_calc
    
    ax.scatter(*pg.T, c="tab:blue", s=80, label="Ground Truth (CoppeliaSim)", depthshade=False)
    ax.scatter(*pc.T, c="tab:red", s=50, marker="^", label="Cálculo Python", depthshade=False)
    
    for a, b in zip(pg, pc):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 
                color="gray", linewidth=0.8, alpha=0.5)
    
    ax.set_title("Dispersão 3D das Posições do Efetuador", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 5 (3D) salvo → {caminho}")


def grafico_erros_por_eixo(erro_pos_x, erro_pos_y, erro_pos_z, caminho):
    """
    Gráfico Novo: Mostra o erro de posição separado por eixo (X, Y, Z).
    Muito útil para identificar em qual direção o erro é maior.
    """
    n = len(erro_pos_x)
    idx = np.arange(1, n + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.25
    ax.bar(idx - width, erro_pos_x * 1000, width=width, label='Erro X (mm)', color='tab:blue', alpha=0.9)
    ax.bar(idx,         erro_pos_y * 1000, width=width, label='Erro Y (mm)', color='tab:green', alpha=0.9)
    ax.bar(idx + width, erro_pos_z * 1000, width=width, label='Erro Z (mm)', color='tab:red', alpha=0.9)
    
    ax.set_title("Erro de Posição por Eixo (X, Y, Z)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Configuração de Teste")
    ax.set_ylabel("Erro (mm)")
    ax.set_xticks(idx)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 6 (Erros por Eixo) salvo → {caminho}")


def imprimir_relatorio(gt, erro_pos, erro_ori):
    """
    Imprime um relatório textual resumido no terminal com as estatísticas 
    de erro mais importantes.
    """
    n = len(gt["thetas"])
    print("\n" + "=" * 70)
    print("  RELATÓRIO — CINEMÁTICA DIRETA (Python vs. CoppeliaSim)")
    print("=" * 70)
    print(f"\n  Configurações testadas: {n}")
    print(f"\n  {'Config':<8}{'ΔPos (mm)':<14}{'ΔOri (Frob)'}")
    print("  " + "-" * 40)
    
    for i in range(n):
        print(f"  C{i + 1:<7}{erro_pos[i] * 1000:<14.4f}{erro_ori[i]:.2e}")
    
    print(f"\n  Erro de posição   — máx: {np.max(erro_pos) * 1000:.4f} mm | médio: {np.mean(erro_pos) * 1000:.4f} mm")
    print(f"  Erro de orientação — máx: {np.max(erro_ori):.2e} | médio: {np.mean(erro_ori):.2e}")
    print("=" * 70 + "\n")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    try:
        robo = InterfaceRoboCoppelia()
    except Exception as erro:
        print(f"\n[ERRO] Não foi possível conectar ao CoppeliaSim: {erro}")
        sys.exit(1)

    gt = robo.coletar_ground_truth()
    robo.salvar_csv(gt, os.path.join(os.path.dirname(__file__), "ground_truth_direta.csv"))

    print("[INFO] Calculando Cinemática Direta em Python...")
    
    # === LINHA ATUALIZADA ===
    erro_pos, erro_pos_x, erro_pos_y, erro_pos_z, erro_ori, pos_calc, rot_calc = avaliar_precisao_direta(gt)

    # Geração dos gráficos (ainda funciona com as variáveis antigas)
    grafico_curvas_juntas(gt, os.path.join(PASTA_SAIDA, "grafico_1_curvas_juntas.png"))
    grafico_curvas_posicao(gt, pos_calc, os.path.join(PASTA_SAIDA, "grafico_2_posicao_garra.png"))
    grafico_curvas_orientacao(gt, rot_calc, os.path.join(PASTA_SAIDA, "grafico_3_orientacao_garra.png"))
    grafico_erros(erro_pos, erro_ori, os.path.join(PASTA_SAIDA, "grafico_4_erros.png"))
    grafico_dispersao_3d(gt, pos_calc, os.path.join(PASTA_SAIDA, "grafico_5_dispersao_3d.png"))

    # === NOVO GRÁFICO ===
    grafico_erros_por_eixo(erro_pos_x, erro_pos_y, erro_pos_z, 
                           os.path.join(PASTA_SAIDA, "grafico_6_erros_por_eixo.png"))

    imprimir_relatorio(gt, erro_pos, erro_ori)

    print(f"\n✅ Validação concluída!")