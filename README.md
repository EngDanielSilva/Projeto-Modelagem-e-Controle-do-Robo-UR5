# UR5 — Robô CoppeliaSim: Cinemática e Planejamento de Trajetória
#   DANIEL SILVA DE SOUZA
#   LUCAS FILAHO XAVIER
Projeto da disciplina de Sistemas Robóticos, dividido nas 3 etapas
propostas: cinemática direta, cinemática inversa e planejamento de
trajetória (pick-and-place do copo). O robô utilizado é o **UR5** com
garra **Robotiq 85**, simulado no **CoppeliaSim**.

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── UR5/                          # arquivos da cena/robô do CoppeliaSim (ver abaixo)
├── cinematica_direta/             # Etapa 1 — cinemática direta do UR5
├── cinematica_inversa/            # Etapa 2 — cinemática inversa do UR5
└── planejamento_trajetoria/       # Etapa 3 — pick-and-place com trajetória cúbica
    ├── modelo_e_trajetoria.py     # cinemática (FK/IK) + geometria da tarefa + polinômio cúbico
    ├── simulacao_e_graficos.py    # comunicação com o CoppeliaSim, validação e gráficos
    ├── main.py                    # script principal — executa o pipeline completo
    └── outputs/                   # figuras geradas (.png) a cada execução
```


## Pasta `UR5/`

Contém os arquivos do robô/cena utilizados pelo **CoppeliaSim**
(software de simulação usado neste projeto):

| Arquivo | Descrição |
|---|---|
| `remApi` | Arquivo de cabeçalho/definições da API remota legada do CoppeliaSim (Remote API). |
| `remoteApi.dll` | Biblioteca compilada da Remote API para Windows. |
| `remoteApi.dylib` | Biblioteca compilada da Remote API para macOS. |
| `remoteApiProto` | Definições de protocolo/funções da Remote API. |
| `UR5` | Cena do CoppeliaSim com o robô UR5 e a garra Robotiq 85 configurados para esta tarefa. |
| `UR5_control` | Script/arquivo auxiliar de controle associado à cena do UR5. |

> Esses arquivos fazem parte da instalação/exemplos do CoppeliaSim e
> são necessários apenas para abrir a cena e estabelecer a comunicação
> remota — não fazem parte do código Python do projeto.

## Requisitos e instalação

- Python 3.9+
- [CoppeliaSim](https://www.coppeliarobotics.com/) instalado, com a
  cena de `UR5/` aberta antes de executar o script (API ZeroMQ
  habilitada).

Instale as dependências Python:

```bash
pip install -r requirements.txt
```

Dependências principais (ver `requirements.txt`):

- `numpy` — álgebra linear (cinemática, Jacobianos, polinômios).
- `matplotlib` — geração dos gráficos.
- `coppeliasim-zmqremoteapi-client` — comunicação com o CoppeliaSim.

## Etapa 3 — Planejamento de Trajetória

Tarefa: o efetuador parte da posição de repouso, pega um copo na
origem, avança a frente, desloca-se lateralmente  e deposita
o copo no destinoe e retorna a origem em um caminho alternativo.

### Como executar

1. Abra o CoppeliaSim com a cena do UR5 (pasta `UR5/`).
2. Rode:

```bash
cd planejamento_trajetoria
python main.py
```

Se o CoppeliaSim não estiver aberto/conectado, o planejamento e os
gráficos de juntas/pose cartesiana ainda são gerados normalmente; só a
execução no simulador e a validação contra o ground truth são
puladas (com aviso no terminal).

### Métodos utilizados

- **Cinemática direta**: cadeia de transformações homogêneas montada a
  partir da pose de repouso de cada elo, lida diretamente da cena.
- **Cinemática inversa**: iterativa, via Jacobiano numérico (diferenças
  finitas) e pseudo-inversa, resolvendo pose completa (posição +
  orientação) para manter o efetuador sempre na mesma orientação
  vertical ao longo da tarefa.
- **Planejamento de trajetória**: polinômio cúbico independente por
  junta, com condições de contorno de posição e velocidade (nula) em
  cada waypoint, resolvido via sistema linear 4×4.
- **Validação**: comparação da pose prevista pela FK (usando os ângulos
  de junta *reais* lidos do simulador) com a pose real do efetuador,
  também lida diretamente do simulador — isola o erro do modelo
  cinemático do erro de rastreamento do controlador de juntas.

### Saídas geradas (`planejamento_trajetoria/outputs/`)

| Arquivo | Conteúdo |
|---|---|
| `pp_juntas.png` | Posição, velocidade e aceleração de cada junta ao longo do tempo. |
| `pp_cartesiano.png` | Posição (x, y, z) e orientação (yaw, pitch, roll) do efetuador. |
| `pp_trajetoria3d.png` | Percurso 3D do efetuador durante a tarefa. |
| `pp_erros_validacao.png` | Erro de posição (mm) e de orientação, comparando a FK com a pose real lida do simulador. |
