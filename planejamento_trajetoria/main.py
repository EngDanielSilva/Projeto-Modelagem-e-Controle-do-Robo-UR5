"""
============================================================
 Etapa A2 - Planejamento de Trajetoria: Pick-and-Place do Copo
 Metodo: Polinomios cubicos por junta (visto em sala)
 Tarefa: origem -> avanco frontal -> deslocamento lateral
         -> destino, com retorno em caminho alternativo

 Script principal -- executa o projeto completo de trajetoria 
============================================================
"""

import os

from modelo_e_trajetoria import (
    ELOS_REPOUSO,
    GARRA_REPOUSO,
    NUM_JUNTAS,
    ModeloCinematicoUR5,
    GeradorTrajetoriaCubica,
    resolver_ik_de_toda_tarefa,
    reconstruir_pose_cartesiana,
)
from simulacao_e_graficos import (
    InterfaceSimulacao,
    montar_agenda_de_eventos,
    executar_tarefa_no_simulador,
    PainelDeGraficos,
)


# ==============================================================================
#  7. ORQUESTRACAO DA EXECUCAO
# ==============================================================================
def montar_plano_de_duracoes():
    """
    10 waypoints (HOME, TRANS_GIRO, TRANS_DESCIDA, PRE_PICK, PICK, PRE_PICK_2,
    AVANCO_FRENTE, PRE_PLACE, PLACE, PRE_PLACE_2, HOME_ALT) => 9 segmentos.
    """
    return [2.5, 2.5, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 5.0]


def main():
    pasta_de_saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(pasta_de_saida, exist_ok=True)
    modelo = ModeloCinematicoUR5(ELOS_REPOUSO, GARRA_REPOUSO)

    print("[INFO] Resolvendo cinematica inversa da sequencia da tarefa...")
    configuracoes_juntas, rotulos = resolver_ik_de_toda_tarefa(modelo)

    print("[INFO] Gerando trajetoria cubica multi-segmento...")
    gerador = GeradorTrajetoriaCubica(NUM_JUNTAS, amostras_por_segmento=150)
    tempo, posicao_juntas, velocidade_juntas, aceleracao_juntas, info = gerador.gerar(
        configuracoes_juntas, montar_plano_de_duracoes()
    )

    posicao_cartesiana, orientacao_rpy = reconstruir_pose_cartesiana(modelo, posicao_juntas)

    try:
        interface = InterfaceSimulacao()
        agenda_eventos = montar_agenda_de_eventos(rotulos, info)
        executar_tarefa_no_simulador(interface, posicao_juntas, agenda_eventos)
        interface.finalizar()
    except Exception as erro:
        print(f"[AVISO] Simulador indisponivel ou interrompido: {erro}")

    print("[INFO] Gerando figuras...")
    painel = PainelDeGraficos()
    painel.grafico_espaco_juntas(
        tempo, posicao_juntas, velocidade_juntas, aceleracao_juntas, info,
        os.path.join(pasta_de_saida, "pp_juntas.png"),
    )
    painel.grafico_espaco_cartesiano(
        tempo, posicao_cartesiana, orientacao_rpy, info,
        os.path.join(pasta_de_saida, "pp_cartesiano.png"),
    )
    painel.grafico_percurso_3d(posicao_cartesiana, os.path.join(pasta_de_saida, "pp_trajetoria3d.png"))

    print("[INFO] Etapa A2 - Pick-and-Place concluido!")


if __name__ == "__main__":
    main()
