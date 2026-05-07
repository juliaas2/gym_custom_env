# Data artifacts

Esta pasta armazena os artefatos gerados pelos experimentos.

- Treinos PPO sao salvos como `data/ppo_cpp_*.zip` ou `data/recurrent_ppo_cpp_*.zip`.
- Treinos Q-learning sao salvos como `data/qlearning_cpp_*.zip`.
- Treinos SARSA sao salvos como `data/sarsa_cpp_*.zip`.

Para a entrega principal, use o modelo Q-learning salvo em `data/qlearning_cpp_*.zip`. Ele contem a Q-table aprendida, hiperparametros e log de treinamento.

Modelo PPO recorrente treinado:

- `recurrent_ppo_cpp_20x20_obs48_view5_steps250000.zip`

Metricas do PPO recorrente:

- `ppo_cpp_20x20_obs48_view5_seed123_episodes50.json`
- `recurrent_ppo_cpp_20x20_obs48_view5_seed123_episodes50_stochastic.json`

Arquivo principal desta entrega:

- `qlearning_cpp_20x20_obs48_view5_seed321_episodes800.zip`

Metricas da avaliacao principal:

- `qlearning_cpp_20x20_obs48_view5_seed123_episodes100.json`
