# Projeto CPP - GridWorld com Q-learning

Links do enunciado:

- https://insper.github.io/rl/classes/22_custom_env/
- https://insper.github.io/rl/classes/23_custom_env/

## Objetivo

O problema implementado e o Coverage Path Planning em um GridWorld com obstaculos. O agente precisa visitar a maior quantidade possivel de celulas livres sem conhecer o mapa completo no inicio do episodio.

Meta usada para nota total:

- 5x5 com 3 obstaculos
- 10x10 com 12 obstaculos
- 20x20 com 48 obstaculos
- cobertura alvo no 20x20: pelo menos 90%

## Representacao de estado

O ambiente retorna:

- `agent`: posicao normalizada `(x / size, y / size)` e cobertura atual.
- `neighbors`: janela local centrada no agente, por padrao `5x5`, com raio 2 em cada direcao.

Na matriz local:

- `0`: celula livre ainda nao visitada
- `1`: obstaculo, parede ou fora do grid
- `2`: celula ja visitada

O agente Q-learning transforma essa observacao local em um estado discreto que inclui:

- janela local `5x5`
- mascara de acoes bloqueadas
- mascara de vizinhos livres ainda nao visitados
- cobertura discretizada
- paridade da linha/coluna
- ultima acao
- uma dica de direcao calculada a partir da memoria local do episodio

Essa memoria e necessaria porque o problema e parcialmente observavel: so a janela local nao informa onde estao as fronteiras descobertas anteriormente.

## Algoritmo

Tambem foi implementado PPO recorrente (`RecurrentPPO`, de `sb3-contrib`) com `MultiInputLstmPolicy`.
O modelo PPO treinado esta salvo em:

- `data/recurrent_ppo_cpp_20x20_obs48_view5_steps250000.zip`

Com o PPO recorrente, a melhor avaliacao obtida em 50 episodios no 20x20 foi:

- modo deterministico: cobertura media 6.60%, 0/50 episodios acima de 90%
- modo estocastico: cobertura media 50.94%, 0/50 episodios acima de 90%

Como esse resultado nao atinge o criterio de 90%, a solucao que bate o criterio usa Q-learning tabular em `utils/cpp_q_learning_agent.py`.

Atualizacao usada:

```text
Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
```

Hiperparametros do treino principal:

- algoritmo: Q-learning
- alpha: 0.25
- gamma: 0.98
- epsilon inicial: 0.80
- epsilon minimo: 0.02
- epsilon decay: 0.992
- episodios de treino: 800
- grid de treino: 20x20
- obstaculos: 48
- max steps: 800
- view size: 3

O modelo salvo contem a Q-table aprendida com 105586 estados.

## Como executar

Treinar novamente:

```bash
python train_grid_world_cpp.py qtrain 20 48 800 800 --algorithm qlearning
```

Testar o modelo salvo:

```bash
python train_grid_world_cpp.py test 20 48 --agent qlearning --model data/qlearning_cpp_20x20_obs48_view5_seed321_episodes800.zip
```

Visualizar um episodio:

```bash
python run_grid_world_cpp.py --size 20 --obstacles 48 --max-steps 800 --render
```

## Artefatos

- Modelo Q-learning: `data/qlearning_cpp_20x20_obs48_view5_seed321_episodes800.zip`
- Metricas de avaliacao: `data/qlearning_cpp_20x20_obs48_view5_seed123_episodes100.json`
- Log de treino: `log/qlearning_cpp_20x20_obs48_view5_seed321_episodes800_training.csv`

## Resultado obtido

Comando usado:

```bash
.venv\Scripts\python.exe train_grid_world_cpp.py test 20 48 --agent qlearning --model data\qlearning_cpp_20x20_obs48_view5_seed321_episodes800.zip --episodes 100 --save-results
```

Resultado:

- Full Coverage Rate: 100.00% (100/100)
- Coverage >= 90% Rate: 100.00% (100/100)
- Average Coverage: 100.00%
- Standard Deviation: 0.00%
- Min Coverage: 100.00%
- Max Coverage: 100.00%
- Average Steps: 513.6
- Min Steps: 477
- Max Steps: 560
- Limite de passos usado no 20x20: 800

Portanto, o criterio de 90%+ no 20x20 foi atingido com folga usando Q-learning e uma Q-table salva em `data/`.
