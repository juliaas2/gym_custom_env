$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"

Write-Host "Stage 1/3 - Recurrent PPO on 5x5"
& $Python train_grid_world_cpp.py train 5 3 200 1000000 `
    --view-size 5 `
    --ppo-kind recurrent `
    --n-envs 4 `
    --n-steps 512 `
    --batch-size 512 `
    --learning-rate 0.0003 `
    --entropy-coef 0.03 `
    --checkpoint-freq 100000 `
    --seed 500 `
    --model-output data\recurrent_ppo_cpp_5x5_obs3_view5_curriculum.zip

Write-Host "Stage 2/3 - Continue on 10x10"
& $Python train_grid_world_cpp.py curriculum 10 12 400 1500000 `
    --view-size 5 `
    --model data\recurrent_ppo_cpp_5x5_obs3_view5_curriculum.zip `
    --n-envs 4 `
    --n-steps 512 `
    --batch-size 512 `
    --learning-rate 0.00025 `
    --entropy-coef 0.02 `
    --checkpoint-freq 100000 `
    --seed 600 `
    --model-output data\recurrent_ppo_cpp_10x10_obs12_view5_curriculum.zip

Write-Host "Stage 3/3 - Continue on 20x20"
& $Python train_grid_world_cpp.py curriculum 20 48 800 3000000 `
    --view-size 5 `
    --model data\recurrent_ppo_cpp_10x10_obs12_view5_curriculum.zip `
    --n-envs 4 `
    --n-steps 512 `
    --batch-size 512 `
    --learning-rate 0.0002 `
    --entropy-coef 0.01 `
    --checkpoint-freq 100000 `
    --seed 700 `
    --model-output data\recurrent_ppo_cpp_20x20_obs48_view5_curriculum.zip

Write-Host "Evaluating final PPO model"
& $Python train_grid_world_cpp.py test 20 48 800 `
    --agent ppo `
    --model data\recurrent_ppo_cpp_20x20_obs48_view5_curriculum.zip `
    --view-size 5 `
    --episodes 100 `
    --save-results `
    --results-path data\recurrent_ppo_cpp_20x20_obs48_view5_curriculum_eval.json

Write-Host "Generating learning curve for latest log"
& $Python plot_cpp_learning.py

Write-Host "Done."
