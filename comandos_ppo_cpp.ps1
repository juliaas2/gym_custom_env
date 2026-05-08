$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"
$TARGET_COVERAGE = 90.0
$TRAIN_ENVS = "5:3:200:1,10:12:500:2,20:48:1000:6"
$TRAIN_COMMON = @(
    "--view-radius", "2",
    "--train-envs", $TRAIN_ENVS,
    "--n-steps", "4096",
    "--batch-size", "512",
    "--gamma", "0.995",
    "--ent-coef", "0.03",
    "--learning-rate", "0.00025",
    "--net-width", "256",
    "--net-layers", "3",
    "--eval-freq", "100000",
    "--eval-episodes-during-train", "30",
    "--seed", "42"
)

function Get-LatestModel($Pattern) {
    $model = Get-ChildItem -Path data -Filter $Pattern | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $model) {
        throw "Modelo nao encontrado com o padrao: $Pattern"
    }
    return $model
}

function Test-SameModel($ModelPath, $Tag) {
    & $PY train_grid_world_cpp.py test 5 3 200 --model-path "$ModelPath" --episodes 100 --run-name "eval5_$Tag" --view-radius 2
    & $PY train_grid_world_cpp.py test 10 12 500 --model-path "$ModelPath" --episodes 100 --run-name "eval10_$Tag" --view-radius 2
    & $PY train_grid_world_cpp.py test 20 48 1000 --model-path "$ModelPath" --episodes 100 --run-name "eval20_$Tag" --view-radius 2
}

function Get-Latest20Coverage($Tag) {
    $summary = Get-ChildItem -Path data -Filter "*eval20_$Tag*_summary.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $summary) {
        return 0.0
    }
    return [double]((Get-Content $summary.FullName | ConvertFrom-Json).average_coverage_percent)
}

& $PY train_grid_world_cpp.py train 20 48 1000 10000000 --run-name ppo_all_sizes_target20 --eval-episodes 100 @TRAIN_COMMON
$MODEL = Get-LatestModel "*ppo_all_sizes_target20_best.zip"
Test-SameModel "$($MODEL.FullName)" "same_model_round0"
$coverage20 = Get-Latest20Coverage "same_model_round0"

for ($round = 1; $round -le 2 -and $coverage20 -lt $TARGET_COVERAGE; $round++) {
    & $PY train_grid_world_cpp.py curriculum 20 48 1000 5000000 --model-path "$($MODEL.FullName)" --run-name "ppo_all_sizes_boost$round" --eval-episodes 100 @TRAIN_COMMON
    $MODEL = Get-LatestModel "*ppo_all_sizes_boost$round*_best.zip"
    Test-SameModel "$($MODEL.FullName)" "same_model_round$round"
    $coverage20 = Get-Latest20Coverage "same_model_round$round"
}

Write-Host "Modelo final unico: $($MODEL.FullName)"
Write-Host "Cobertura media final no 20x20: $coverage20%"
