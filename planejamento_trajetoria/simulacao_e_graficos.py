"""
============================================================
 Etapa A2 - Planejamento de Trajetoria: Pick-and-Place do Copo
 Metodo: Polinomios cubicos por junta (visto em sala)

 Modulo: comunicacao com o CoppeliaSim, validacao e graficos
============================================================
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERRO] Instale a biblioteca: pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)

from modelo_e_trajetoria import (
    NUM_JUNTAS,
    PONTO_ORIGEM_COPO_MUNDO,
    PONTO_DESTINO_COPO_MUNDO,
    CAMINHO_OBJ_COPO,
    CAMINHO_PONTO_FIXACAO,
    CAMINHO_JUNTA_DIREITA_GARRA,
    CAMINHO_JUNTA_ESQUERDA_GARRA,
)


# ==============================================================================
#  5. INTERFACE COM O COPPELIASIM
# ==============================================================================
class InterfaceSimulacao:
    """Encapsula a comunicacao ZMQ com o CoppeliaSim: juntas, garra e objeto manipulado."""

    def __init__(self, host="localhost", porta=23000):
        print(f"[SIM] Conectando ao CoppeliaSim ({host}:{porta})...")
        self.client = RemoteAPIClient(host=host, port=porta)
        self.sim = self.client.require("sim")
        self.handles_juntas = [self.sim.getObject(f"/UR5_joint{i}") for i in range(1, 7)]
        self.client.setStepping(True)
        self.sim.startSimulation()
        print("[SIM] Conectado. Modo sincrono ativo.")

    def finalizar(self):
        self.client.setStepping(False)
        self.sim.stopSimulation()

    def avancar_passo_fisico(self):
        self.client.step()

    def enviar_alvo_juntas(self, juntas_rad):
        for handle, valor in zip(self.handles_juntas, juntas_rad):
            self.sim.setJointTargetPosition(handle, float(valor))

    def esperar_chegada(self, juntas_alvo_rad, tempo_limite_s=1.5, tolerancia_rad=2e-3):
        alvo = np.asarray(juntas_alvo_rad, dtype=float)
        instante_inicial = time.time()
        while (time.time() - instante_inicial) < tempo_limite_s:
            atuais = np.array([self.sim.getJointPosition(h) for h in self.handles_juntas])
            if np.max(np.abs(atuais - alvo)) < tolerancia_rad:
                return True
            self.avancar_passo_fisico()
        return False

    def _comandar_garra(self, fechar, deslocamento=0.45):
        try:
            handle_dir = self.sim.getObject(CAMINHO_JUNTA_DIREITA_GARRA)
            handle_esq = self.sim.getObject(CAMINHO_JUNTA_ESQUERDA_GARRA)
            pos_dir = self.sim.getJointPosition(handle_dir)
            pos_esq = self.sim.getJointPosition(handle_esq)
            sinal = 1.0 if fechar else -1.0
            self.sim.setJointTargetPosition(handle_dir, pos_dir + sinal * deslocamento)
            self.sim.setJointTargetPosition(handle_esq, pos_esq - sinal * deslocamento)
        except Exception as erro:
            if fechar:
                print(f"[AVISO] Garra nao encontrada: {erro}")

    def anexar_objeto(self, caminho_objeto, caminho_ponto_fixacao):
        objeto = self.sim.getObject(caminho_objeto)
        ponto_fixacao = self.sim.getObject(caminho_ponto_fixacao)
        self.sim.setObjectInt32Param(objeto, self.sim.shapeintparam_static, 0)
        self.sim.setObjectParent(objeto, ponto_fixacao, True)
        self._comandar_garra(fechar=True)
        print("  >> Objeto anexado a garra.")

    def soltar_objeto(self, caminho_objeto, posicao_destino_mundo=None, orientacao_destino_mundo=None):
        objeto = self.sim.getObject(caminho_objeto)
        self.sim.setObjectInt32Param(objeto, self.sim.shapeintparam_static, 1)
        self._comandar_garra(fechar=False)
        self.sim.setObjectParent(objeto, -1, True)
        if posicao_destino_mundo is not None:
            self.sim.setObjectPosition(objeto, -1, [float(v) for v in posicao_destino_mundo])
        if orientacao_destino_mundo is not None:
            self.sim.setObjectQuaternion(objeto, -1, [float(v) for v in orientacao_destino_mundo])
        self.sim.setObjectInt32Param(objeto, self.sim.shapeintparam_respondable, 0)
        print("  >> Objeto solto na pose de destino.")


def montar_agenda_de_eventos(rotulos_sequencia, info_trajetoria):
    """
    Em vez de comparar o indice da amostra atual com dois indices fixos
    (pick/place) dentro do loop de execucao, pre-calculamos um dicionario
    {indice_da_amostra: nome_do_evento} e o loop principal so consulta esse
    dicionario a cada passo.
    """
    agenda = {}
    indice_chegada_pick = info_trajetoria["marcas_indice"][rotulos_sequencia.index("PICK")] - 1
    indice_chegada_place = info_trajetoria["marcas_indice"][rotulos_sequencia.index("PLACE")] - 1
    agenda[indice_chegada_pick] = "PEGAR"
    agenda[indice_chegada_place] = "SOLTAR"
    return agenda


def executar_tarefa_no_simulador(interface: InterfaceSimulacao, trajetoria_juntas, agenda_eventos):
    sim = interface.sim
    objeto_copo = sim.getObject(CAMINHO_OBJ_COPO)

    orientacao_original_copo = sim.getObjectQuaternion(objeto_copo, -1)
    sim.setObjectPosition(objeto_copo, -1, [float(v) for v in PONTO_ORIGEM_COPO_MUNDO])
    sim.setObjectQuaternion(objeto_copo, -1, [float(v) for v in orientacao_original_copo])
    sim.setObjectInt32Param(objeto_copo, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(objeto_copo, sim.shapeintparam_respondable, 0)

    total_amostras = len(trajetoria_juntas)
    print(f"\n  [Execucao] Pick-and-place ({total_amostras} pontos)...")

    for indice_amostra, juntas in enumerate(trajetoria_juntas):
        interface.enviar_alvo_juntas(juntas)
        interface.avancar_passo_fisico()

        evento = agenda_eventos.get(indice_amostra)
        if evento is not None:
            interface.esperar_chegada(juntas)
            if evento == "PEGAR":
                interface.anexar_objeto(CAMINHO_OBJ_COPO, CAMINHO_PONTO_FIXACAO)
            elif evento == "SOLTAR":
                interface.soltar_objeto(CAMINHO_OBJ_COPO, PONTO_DESTINO_COPO_MUNDO, orientacao_original_copo)
            for _ in range(20):
                interface.avancar_passo_fisico()

        if indice_amostra % 100 == 0:
            print(f"    Ponto {indice_amostra + 1}/{total_amostras}")

    print("  [Execucao] Concluido!\n")


# ==============================================================================
#  6. VISUALIZACAO DOS RESULTADOS
# ==============================================================================
class PainelDeGraficos:
    cores_por_junta = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    rotulos_juntas = [f'Junta {i + 1}' for i in range(NUM_JUNTAS)]

    def grafico_espaco_juntas(self, tempo, posicao, velocidade, aceleracao, info, caminho_arquivo=None):
        figura, eixos = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        figura.suptitle(
            "Pick-and-Place do Copo - Trajetoria Cubica (avanco frontal + espelhamento)",
            fontsize=13, fontweight='bold',
        )

        series = [
            (eixos[0], np.degrees(posicao), "Posicao (graus)"),
            (eixos[1], np.degrees(velocidade), "Velocidade (graus/s)"),
            (eixos[2], np.degrees(aceleracao), "Aceleracao (graus/s^2)"),
        ]
        for eixo, valores, rotulo_eixo_y in series:
            for indice_junta in range(NUM_JUNTAS):
                eixo.plot(
                    tempo, valores[:, indice_junta],
                    color=self.cores_por_junta[indice_junta],
                    label=self.rotulos_juntas[indice_junta], lw=1.6,
                )
            for marca in info["marcas_tempo"]:
                eixo.axvline(marca, color='gray', linestyle=':', lw=0.9)
            eixo.set_ylabel(rotulo_eixo_y)
            eixo.grid(True, alpha=0.35)

        eixos[0].legend(fontsize=8, ncol=6, loc='lower right')
        eixos[2].set_xlabel("Tempo (s)")
        plt.tight_layout()
        self._finalizar_figura(caminho_arquivo)

    def grafico_espaco_cartesiano(self, tempo, posicao, orientacao_rpy, info, caminho_arquivo=None):
        figura, eixos = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

        cores_posicao = ['#e41a1c', '#377eb8', '#4daf4a']
        for indice, rotulo in enumerate(['X (m)', 'Y (m)', 'Z (m)']):
            eixos[0].plot(tempo, posicao[:, indice], color=cores_posicao[indice], label=rotulo, lw=2)
        eixos[0].set_ylabel("Posicao (m)"); eixos[0].legend(); eixos[0].grid(True, alpha=0.35)

        cores_orientacao = ['#984ea3', '#ff7f00', '#a65628']
        for indice, rotulo in enumerate(['Yaw (graus)', 'Pitch (graus)', 'Roll (graus)']):
            eixos[1].plot(tempo, np.degrees(orientacao_rpy[:, indice]), color=cores_orientacao[indice], label=rotulo, lw=2)
        eixos[1].set_ylabel("Orientacao (graus)"); eixos[1].set_xlabel("Tempo (s)")
        eixos[1].legend(); eixos[1].grid(True, alpha=0.35)

        plt.tight_layout()
        self._finalizar_figura(caminho_arquivo)

    def grafico_percurso_3d(self, posicao, caminho_arquivo=None):
        figura = plt.figure(figsize=(9, 7))
        eixo = figura.add_subplot(111, projection='3d')
        eixo.plot(posicao[:, 0], posicao[:, 1], posicao[:, 2], color='steelblue', lw=2)
        eixo.set_xlabel("X (m)"); eixo.set_ylabel("Y (m)"); eixo.set_zlabel("Z (m)")
        eixo.grid(True, alpha=0.3)
        plt.tight_layout()
        self._finalizar_figura(caminho_arquivo)

    @staticmethod
    def _finalizar_figura(caminho_arquivo):
        if caminho_arquivo:
            plt.savefig(caminho_arquivo, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
