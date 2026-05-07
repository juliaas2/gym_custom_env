# Comandos para treinar PPO com curriculo

O PPO usado e o `RecurrentPPO` do pacote `sb3-contrib`, com observacao parcial `5x5`.

## Rodar tudo de uma vez

No PowerShell, dentro da pasta `gym_custom_env`, rode:

```powershell
.\train_ppo_curriculum.ps1
```

Se o PowerShell bloquear script local, rode uma vez:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\train_ppo_curriculum.ps1
```

## Etapas que o script executa

1. Treina no `5x5`, 3 obstaculos, 1.000.000 timesteps.
2. Continua no `10x10`, 12 obstaculos, 1.500.000 timesteps.
3. Continua no `20x20`, 48 obstaculos, 3.000.000 timesteps.
4. Avalia o modelo final em 100 episodios.
5. Gera um grafico de aprendizagem com `plot_cpp_learning.py`.

## Arquivos principais

Modelo final esperado:

```text
data/recurrent_ppo_cpp_20x20_obs48_view5_curriculum.zip
```

Resultado da avaliacao:

```text
data/recurrent_ppo_cpp_20x20_obs48_view5_curriculum_eval.json
```

Grafico:

```text
log/<ultima_pasta_de_treino>/learning_curve.png
```

## Se nao bater 90%

Continue o treino do `20x20`:

```powershell
.\.venv\Scripts\python.exe train_grid_world_cpp.py curriculum 20 48 800 1000000 `
  --view-size 5 `
  --model data\recurrent_ppo_cpp_20x20_obs48_view5_curriculum.zip `
  --n-envs 4 `
  --n-steps 512 `
  --batch-size 512 `
  --learning-rate 0.0001 `
  --entropy-coef 0.005 `
  --checkpoint-freq 100000 `
  --model-output data\recurrent_ppo_cpp_20x20_obs48_view5_curriculum.zip
```

Depois avalie de novo:

```powershell
.\.venv\Scripts\python.exe train_grid_world_cpp.py test 20 48 800 `
  --agent ppo `
  --model data\recurrent_ppo_cpp_20x20_obs48_view5_curriculum.zip `
  --view-size 5 `
  --episodes 100 `
  --save-results `
  --results-path data\recurrent_ppo_cpp_20x20_obs48_view5_curriculum_eval.json
```
