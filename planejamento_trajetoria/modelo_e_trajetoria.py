"""
============================================================
 Etapa A2 - Planejamento de Trajetoria: Pick-and-Place do Copo
 Metodo: Polinomios cubicos por junta (visto em sala)
 Tarefa: origem -> avanco frontal -> deslocamento lateral (espelhado)
         -> destino, com retorno em postura alternativa

 Modulo: cinematica (FK/IK) + geometria da tarefa + polinomio cubico
============================================================
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

NUM_JUNTAS = 6


# ==============================================================================
#  1. CALIBRACAO DA CENA (geometria fisica do UR5 nesta cena do CoppeliaSim)
# ==============================================================================
ELOS_REPOUSO = [
    ([0.0,       0.0,      0.0085  ], [0.0,      0.0,       0.0,      1.0     ]),
    ([-0.070317, -1.4e-05, 0.066039], [6.8e-05, -0.707159,  6.8e-05,  0.707054]),
    ([0.425105,  -9.1e-05, 1.9e-05 ], [-0.0,    -1.5e-05,  -0.000107, 1.0     ]),
    ([0.392149,  -7.6e-05, 2e-06   ], [-0.0,    -0.0,      -9.7e-05,  1.0     ]),
    ([0.045573,  -1e-05,   0.03971 ], [7.5e-05,  0.707098, -7.5e-05,  0.707116]),
    ([0.014424,  -4e-06,   0.049176], [-8.5e-05,-0.707107, -8.5e-05,  0.707107]),
]
GARRA_REPOUSO = ([-0.0, -0.0, 0.089375], [-0.0, -0.0, -9.7e-05, 1.0])


# ==============================================================================
#  2. MODELO CINEMATICO (classe propria: FK + IK numerica)
# ==============================================================================
class ModeloCinematicoUR5:
    """Cinematica direta e inversa numerica do UR5, a partir da cadeia de elos da cena."""

    def __init__(self, elos, garra):
        self.elos = elos
        self.garra = garra

    @staticmethod
    def quat_para_matriz(q):
        qx, qy, qz, qw = q
        return np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw),     1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx**2 + qy**2)],
        ])

    @staticmethod
    def matriz_rotacao_z(angulo):
        c, s = np.cos(angulo), np.sin(angulo)
        T = np.eye(4)
        T[0, 0], T[0, 1] = c, -s
        T[1, 0], T[1, 1] = s, c
        return T

    def _homogenea(self, pos, quat):
        T = np.eye(4)
        T[:3, :3] = self.quat_para_matriz(quat)
        T[:3, 3] = pos
        return T

    def cinematica_direta(self, juntas):
        """Devolve a matriz homogenea 4x4 do efetuador para um vetor de juntas (rad)."""
        T = np.eye(4)
        for indice_elo, (pos, quat) in enumerate(self.elos):
            T = T @ self._homogenea(pos, quat) @ self.matriz_rotacao_z(juntas[indice_elo])
        return T @ self._homogenea(*self.garra)

    def posicao_ponta(self, juntas, offset_ao_longo_z=0.0):
        T = self.cinematica_direta(juntas)
        return T[:3, 3] + offset_ao_longo_z * T[:3, 2]

    def jacobiano_posicao(self, juntas, passo=1e-6):
        """
        Jacobiano 3x6 (somente posicao) do FLANGE (sem offset), por diferencas
        progressivas. Usado como direcao de busca da IK -- e uma aproximacao
        (a derivada exata do ponto de fixacao real seria ligeiramente
        diferente, pois o offset gira junto com o punho), mas e a mesma
        aproximacao usada no calculo original/validado, e mante-la e o que
        faz a IK convergir para a MESMA familia de solucoes (cotovelo/punho)
        testada em simulacao -- ou seja, abordagem de aproximacao quase
        vertical sobre o copo, em vez de uma solucao lateral alternativa.
        """
        T_base = self.cinematica_direta(juntas)
        p_base = T_base[:3, 3]
        colunas = []
        for indice in range(NUM_JUNTAS):
            juntas_perturbadas = juntas.copy()
            juntas_perturbadas[indice] += passo
            p_perturbado = self.cinematica_direta(juntas_perturbadas)[:3, 3]
            colunas.append((p_perturbado - p_base) / passo)
        return np.column_stack(colunas)

    @staticmethod
    def erro_orientacao(R_atual, R_alvo):
        """
        Erro de orientacao (vetor 3D) entre duas matrizes de rotacao, pelo
        metodo classico (soma dos produtos vetoriais entre colunas
        correspondentes de R_atual e R_alvo -- ver Siciliano, "Robotics").
        O vetor vale ~0 quando as orientacoes coincidem e cresce suavemente
        com o desalinhamento, funcionando como termo de erro angular para a
        IK de pose completa (posicao + orientacao).
        """
        erro = np.zeros(3)
        for indice_coluna in range(3):
            erro += np.cross(R_atual[:, indice_coluna], R_alvo[:, indice_coluna])
        return 0.5 * erro

    def jacobiano_pose(self, juntas, offset_ao_longo_z=0.0, passo=1e-6):
        """
        Jacobiano 6xN (posicao + orientacao) do ponto de fixacao da garra,
        por diferencas progressivas. As 3 primeiras linhas sao a derivada da
        posicao (mesma ideia de 'jacobiano_posicao', incluindo o offset). As
        3 ultimas sao a derivada da orientacao: a partir da parte
        antissimetrica de R(juntas+passo) @ R(juntas).T -- uma rotacao
        infinitesimal entre as duas poses -- extraimos o vetor de velocidade
        angular equivalente (operador 'vee') e dividimos pelo passo.
        """
        T_base = self.cinematica_direta(juntas)
        R_base = T_base[:3, :3]
        p_base = T_base[:3, 3] + offset_ao_longo_z * T_base[:3, 2]
        colunas_pos, colunas_orient = [], []
        for indice in range(NUM_JUNTAS):
            juntas_perturbadas = juntas.copy()
            juntas_perturbadas[indice] += passo
            T_perturbado = self.cinematica_direta(juntas_perturbadas)
            R_perturbado = T_perturbado[:3, :3]
            p_perturbado = T_perturbado[:3, 3] + offset_ao_longo_z * T_perturbado[:3, 2]
            colunas_pos.append((p_perturbado - p_base) / passo)

            R_delta = R_perturbado @ R_base.T
            antissimetrica = (R_delta - R_delta.T) / 2.0
            vetor_w = np.array([antissimetrica[2, 1], antissimetrica[0, 2], antissimetrica[1, 0]])
            colunas_orient.append(vetor_w / passo)
        return np.vstack([np.column_stack(colunas_pos), np.column_stack(colunas_orient)])

    def resolver_ik(self, posicao_desejada, chute_inicial, max_iteracoes=500,
                     ganho=0.5, tolerancia_m=1e-7, offset_ao_longo_z=0.0,
                     orientacao_desejada=None, tolerancia_rad=1e-6):
        """
        Busca iterativa (jacobiano + pseudo-inversa) das juntas que levam o
        PONTO DE FIXACAO DA GARRA (flange + offset_ao_longo_z ao longo do
        eixo Z do efetuador) para 'posicao_desejada'.

        Se 'orientacao_desejada' (matriz de rotacao 3x3) for informada, a IK
        passa a resolver a POSE COMPLETA (posicao + orientacao): o erro de
        orientacao entra no vetor de erro e o Jacobiano usado e' o 6xN
        ('jacobiano_pose'). Com 6 equacoes (3 de posicao + 3 de orientacao)
        para 6 juntas, a solucao deixa de ser redundante -- antes, com
        apenas 3 equacoes de posicao para 6 juntas, a orientacao do
        efetuador "sobrava" livre e o solver convergia para qualquer
        orientacao proxima do chute inicial (causa da garra chegar torta).
        Agora a garra converge sempre para a MESMA orientacao alvo (ex.:
        sempre vertical, apontando para baixo) em todos os waypoints.

        Sem 'orientacao_desejada', mantem o comportamento antigo (somente
        posicao), por compatibilidade. Retorna
        (juntas_encontradas, erro_pos_residual_m, erro_orient_residual_rad).
        """
        juntas = chute_inicial.copy()

        def calcular_erros(j):
            erro_pos = posicao_desejada - self.posicao_ponta(j, offset_ao_longo_z)
            if orientacao_desejada is None:
                return erro_pos, None
            R_atual = self.cinematica_direta(j)[:3, :3]
            return erro_pos, self.erro_orientacao(R_atual, orientacao_desejada)

        erro_pos, erro_orient = calcular_erros(juntas)
        iteracao = 0
        while iteracao < max_iteracoes:
            convergiu_pos = np.linalg.norm(erro_pos) < tolerancia_m
            convergiu_orient = orientacao_desejada is None or np.linalg.norm(erro_orient) < tolerancia_rad
            if convergiu_pos and convergiu_orient:
                break

            if orientacao_desejada is None:
                J = self.jacobiano_posicao(juntas)
                erro = erro_pos
            else:
                J = self.jacobiano_pose(juntas, offset_ao_longo_z=offset_ao_longo_z)
                erro = np.concatenate([erro_pos, erro_orient])

            juntas = juntas + ganho * np.linalg.pinv(J) @ erro
            juntas = (juntas + np.pi) % (2 * np.pi) - np.pi
            erro_pos, erro_orient = calcular_erros(juntas)
            iteracao += 1

        erro_pos_norm = float(np.linalg.norm(erro_pos))
        erro_orient_norm = float(np.linalg.norm(erro_orient)) if erro_orient is not None else 0.0
        return juntas, erro_pos_norm, erro_orient_norm


# ==============================================================================
#  3. GEOMETRIA DA TAREFA
# ==============================================================================
GRAUS_POSTURA_HOME = np.array([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
JUNTAS_HOME = np.radians(GRAUS_POSTURA_HOME)
JUNTAS_REF_ABAIXADA = np.radians([0.0, 0.0, -90.0, 0.0, 90.0, 0.0])

# Postura final alternativa de retorno: cotovelo/punho iguais a Home, base +90 graus.
GRAUS_POSTURA_HOME_ALT = np.array([90.0, -90.0, 90.0, -90.0, -90.0, 0.0])
JUNTAS_HOME_ALT = np.radians(GRAUS_POSTURA_HOME_ALT)

POSICAO_BASE_MUNDO = np.array([-0.214, -0.314, 0.4246])
ORIENTACAO_BASE_MUNDO = np.array([0.0, 0.0, 1.0, 0.0])
_MATRIZ_BASE_MUNDO = ModeloCinematicoUR5.quat_para_matriz(ORIENTACAO_BASE_MUNDO)


def mundo_para_referencial_base(ponto_mundo):
    return _MATRIZ_BASE_MUNDO.T @ (np.asarray(ponto_mundo, dtype=float) - POSICAO_BASE_MUNDO)


# Pontos de interesse da tarefa (mundo). Destino: espelhado em Y + avancado em X.
PONTO_ORIGEM_COPO_MUNDO = np.array([0.400, -0.358, 0.4717])
DISTANCIA_AVANCO_X_M = 0.15
DISTANCIA_DESLOC_Y_M = 0.208
PONTO_DESTINO_COPO_MUNDO = PONTO_ORIGEM_COPO_MUNDO + np.array(
    [DISTANCIA_AVANCO_X_M, -DISTANCIA_DESLOC_Y_M, 0.0]
)

# ALTURA_PEGA_ACIMA_DA_BASE_M: o quanto, acima do pivot/base do copo
# (PONTO_ORIGEM/DESTINO_COPO_MUNDO), a garra deve fechar. O pivot do copo
# normalmente fica perto da BASE dele; se a garra for direto pra essa altura,
# o corpo da garra acaba descendo demais e "engolindo" o copo (ele fica por
# dentro/atravessado). Subindo esse valor, a garra passa a fechar mais perto
# da regiao da parede/boca do copo, em vez da base.
# AJUSTE FINO: se ainda entrar muito no copo, AUMENTE este valor; se a garra
# fechar no ar acima do copo (sem tocar), DIMINUA. O copo nesta cena nao foi
# medido remotamente, este e' um ponto de partida razoavel.
ALTURA_PEGA_ACIMA_DA_BASE_M = 0.09 # aqui ajustei manualmente ate ficar na altrua correta do copo

PONTO_ORIGEM_PEGA_MUNDO = PONTO_ORIGEM_COPO_MUNDO + np.array([0.0, 0.0, ALTURA_PEGA_ACIMA_DA_BASE_M])
PONTO_DESTINO_PEGA_MUNDO = PONTO_DESTINO_COPO_MUNDO + np.array([0.0, 0.0, ALTURA_PEGA_ACIMA_DA_BASE_M])

# IMPORTANTE: a IK (construir_pontos_da_tarefa) usa os pontos de PEGA (com a
# altura ja ajustada) -- e' so para onde a garra se move. Os pontos _COPO_MUNDO
# originais continuam sendo usados para POSICIONAR O OBJETO em si na cena
# (executar_tarefa_no_simulador / soltar_objeto), que nao devem mudar.
PONTO_ORIGEM_COPO = mundo_para_referencial_base(PONTO_ORIGEM_PEGA_MUNDO)
PONTO_DESTINO_COPO = mundo_para_referencial_base(PONTO_DESTINO_PEGA_MUNDO)

ALTURA_DE_TRANSPORTE_M = 0.12
ALCANCE_GARRA_M = 0.084

CAMINHO_OBJ_COPO = "/Cup"
CAMINHO_PONTO_FIXACAO = "/UR5/ROBOTIQ_85/ROBOTIQ_85_attachPoint"
CAMINHO_JUNTA_DIREITA_GARRA = "/UR5/ROBOTIQ_85/ROBOTIQ_85_RactiveJoint"
CAMINHO_JUNTA_ESQUERDA_GARRA = "/UR5/ROBOTIQ_85/ROBOTIQ_85_LactiveJoint"


@dataclass
class PontoDeApoio:
    """Um waypoint da tarefa: nome descritivo + posicao cartesiana alvo (ref. base)."""
    rotulo: str
    posicao_alvo: np.ndarray
    juntas_resolvidas: Optional[np.ndarray] = None


def construir_pontos_da_tarefa():
    """Devolve, em ordem, os PontoDeApoio que descrevem o pick-and-place."""
    deslocamento_altura = np.array([0.0, 0.0, ALTURA_DE_TRANSPORTE_M])

    acima_origem = PONTO_ORIGEM_COPO + deslocamento_altura
    avanco_frontal = np.array([
        PONTO_DESTINO_COPO[0],
        PONTO_ORIGEM_COPO[1],
        PONTO_ORIGEM_COPO[2] + ALTURA_DE_TRANSPORTE_M,
    ])
    acima_destino = PONTO_DESTINO_COPO + deslocamento_altura

    return [
        PontoDeApoio("PRE_PICK", acima_origem),
        PontoDeApoio("PICK", PONTO_ORIGEM_COPO),
        PontoDeApoio("PRE_PICK_2", acima_origem.copy()),
        PontoDeApoio("AVANCO_FRENTE", avanco_frontal),
        PontoDeApoio("PRE_PLACE", acima_destino),
        PontoDeApoio("PLACE", PONTO_DESTINO_COPO),
        PontoDeApoio("PRE_PLACE_2", acima_destino.copy()),
    ]


def calcular_posturas_de_transicao(juntas_pre_pick):
    """
    Duas posturas intermediarias entre Home e PRE_PICK, para o robo nao saltar
    direto (evita 'varrer' a mesa com o braco aberto):
      - gira so a base (J1) na direcao do copo, braco ainda em Home;
      - depois interpola na metade do caminho rumo a postura final de PRE_PICK.
    """
    postura_base_girada = JUNTAS_HOME.copy()
    postura_base_girada[0] = juntas_pre_pick[0]
    postura_intermediaria = (postura_base_girada + juntas_pre_pick) / 2.0
    return postura_base_girada, postura_intermediaria


def _verificar_convergencia(rotulo, erro_m, erro_rad):
    """Levanta erro se a IK de pose nao convergiu dentro de uma tolerancia pratica."""
    if erro_m * 1000 > 1.0 or np.degrees(erro_rad) > 0.5:
        raise RuntimeError(
            f"IK nao convergiu para '{rotulo}' "
            f"(erro_pos={erro_m*1000:.2f} mm, erro_orient={np.degrees(erro_rad):.3f} graus)"
        )


def resolver_ik_de_toda_tarefa(modelo: ModeloCinematicoUR5):
    """
    Percorre os pontos da tarefa resolvendo a IK em sequencia (cada solucao usa
    a anterior como chute inicial). Devolve a lista de configuracoes de juntas
    e a lista de rotulos correspondente, incluindo HOME no inicio e HOME_ALT no fim.

    A orientacao alvo da garra em TODOS os waypoints e' travada na mesma
    orientacao vertical (eixo Z do efetuador apontando para baixo) usada na
    postura de referencia JUNTAS_REF_ABAIXADA. Antes, a IK resolvia so a
    posicao (3 equacoes para 6 juntas), e a orientacao final variava
    livremente conforme o chute inicial -- por isso a garra chegava torta,
    em vez de descer reta sobre o copo.
    """
    pontos = construir_pontos_da_tarefa()
    orientacao_vertical = modelo.cinematica_direta(JUNTAS_REF_ABAIXADA)[:3, :3]

    juntas_pre_pick, erro_m, erro_rad = modelo.resolver_ik(
        pontos[0].posicao_alvo, JUNTAS_REF_ABAIXADA.copy(),
        offset_ao_longo_z=ALCANCE_GARRA_M, orientacao_desejada=orientacao_vertical,
    )
    _verificar_convergencia(pontos[0].rotulo, erro_m, erro_rad)

    giro_base, meio_caminho = calcular_posturas_de_transicao(juntas_pre_pick)

    rotulos = ["HOME", "TRANS_GIRO", "TRANS_DESCIDA", pontos[0].rotulo]
    configuracoes = [JUNTAS_HOME, giro_base, meio_caminho, juntas_pre_pick]

    chute_atual = juntas_pre_pick
    for ponto in pontos[1:]:
        juntas, erro_m, erro_rad = modelo.resolver_ik(
            ponto.posicao_alvo, chute_atual.copy(),
            offset_ao_longo_z=ALCANCE_GARRA_M, orientacao_desejada=orientacao_vertical,
        )
        _verificar_convergencia(ponto.rotulo, erro_m, erro_rad)
        configuracoes.append(juntas)
        rotulos.append(ponto.rotulo)
        chute_atual = juntas

    configuracoes.append(JUNTAS_HOME_ALT)
    rotulos.append("HOME_ALT")
    return configuracoes, rotulos


# ==============================================================================
#  4. GERADOR DE TRAJETORIA -- POLINOMIO CUBICO POR JUNTA
# ==============================================================================
class SegmentoCubico:
    """
    Representa o polinomio cubico q(t) = c0 + c1*t + c2*t^2 + c3*t^3 de UMA
    junta, entre dois instantes de fronteira, com velocidade inicial/final
    impostas (tipicamente zero, parada em cada waypoint).

    Os coeficientes sao obtidos resolvendo o sistema linear das 4 condicoes
    de contorno (posicao e velocidade em t=0 e t=T), em vez de aplicar a
    formula fechada diretamente -- mesma matematica do polinomio cubico
    visto em sala, escrita de outra forma.
    """

    def __init__(self, posicao_inicial, posicao_final, duracao,
                 velocidade_inicial=0.0, velocidade_final=0.0):
        self.q0 = posicao_inicial
        self.qf = posicao_final
        self.T = duracao
        self.v0 = velocidade_inicial
        self.vf = velocidade_final
        self.coeficientes = self._resolver_coeficientes()

    def _resolver_coeficientes(self):
        T = self.T
        matriz_condicoes = np.array([
            [1.0, 0.0,     0.0,       0.0],
            [0.0, 1.0,     0.0,       0.0],
            [1.0, T,       T ** 2,    T ** 3],
            [0.0, 1.0,     2 * T,     3 * T ** 2],
        ])
        vetor_condicoes = np.array([self.q0, self.v0, self.qf, self.vf])
        return np.linalg.solve(matriz_condicoes, vetor_condicoes)

    def amostrar(self, instantes):
        c0, c1, c2, c3 = self.coeficientes
        posicao = c0 + c1 * instantes + c2 * instantes**2 + c3 * instantes**3
        velocidade = c1 + 2 * c2 * instantes + 3 * c3 * instantes**2
        aceleracao = 2 * c2 + 6 * c3 * instantes
        return posicao, velocidade, aceleracao


class GeradorTrajetoriaCubica:
    """Encadeia SegmentoCubico junta-a-junta ao longo de uma sequencia de waypoints."""

    def __init__(self, num_juntas, amostras_por_segmento=150):
        self.num_juntas = num_juntas
        self.amostras_por_segmento = amostras_por_segmento

    def gerar(self, sequencia_juntas, duracoes_segmentos):
        blocos_tempo, blocos_pos, blocos_vel, blocos_acel = [], [], [], []
        tempo_acumulado = 0.0
        marcas_tempo = [0.0]
        marcas_indice = [0]

        total_segmentos = len(sequencia_juntas) - 1
        for indice_segmento in range(total_segmentos):
            config_inicial = sequencia_juntas[indice_segmento]
            config_final = sequencia_juntas[indice_segmento + 1]
            duracao = duracoes_segmentos[indice_segmento]
            instantes_locais = np.linspace(0.0, duracao, self.amostras_por_segmento)

            pos_segmento = np.empty((self.amostras_por_segmento, self.num_juntas))
            vel_segmento = np.empty_like(pos_segmento)
            acel_segmento = np.empty_like(pos_segmento)

            for junta in range(self.num_juntas):
                segmento = SegmentoCubico(config_inicial[junta], config_final[junta], duracao)
                pos_segmento[:, junta], vel_segmento[:, junta], acel_segmento[:, junta] = \
                    segmento.amostrar(instantes_locais)

            blocos_tempo.append(instantes_locais + tempo_acumulado)
            blocos_pos.append(pos_segmento)
            blocos_vel.append(vel_segmento)
            blocos_acel.append(acel_segmento)

            tempo_acumulado += duracao
            marcas_tempo.append(tempo_acumulado)
            marcas_indice.append(marcas_indice[-1] + self.amostras_por_segmento)

        info = {
            "marcas_tempo": marcas_tempo,
            "marcas_indice": marcas_indice,
            "amostras_por_segmento": self.amostras_por_segmento,
        }
        return (
            np.concatenate(blocos_tempo),
            np.concatenate(blocos_pos, axis=0),
            np.concatenate(blocos_vel, axis=0),
            np.concatenate(blocos_acel, axis=0),
            info,
        )


def reconstruir_pose_cartesiana(modelo: ModeloCinematicoUR5, trajetoria_juntas):
    total_amostras = trajetoria_juntas.shape[0]
    posicoes = np.zeros((total_amostras, 3))
    orientacoes_rpy = np.zeros((total_amostras, 3))
    for indice in range(total_amostras):
        T = modelo.cinematica_direta(trajetoria_juntas[indice])
        posicoes[indice] = T[:3, 3]
        R = T[:3, :3]
        orientacoes_rpy[indice, 1] = np.arcsin(np.clip(-R[2, 0], -1, 1))
        orientacoes_rpy[indice, 0] = np.arctan2(R[1, 0], R[0, 0])
        orientacoes_rpy[indice, 2] = np.arctan2(R[2, 1], R[2, 2])
    return posicoes, orientacoes_rpy
