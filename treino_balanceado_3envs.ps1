$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"
$TARGET_COVERAGE = 90.0

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "TREINO AGRESSIVO - 1 MODELO" -ForegroundColor Green
Write-Host "OBJETIVO: >90% EM TODOS!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# ETAPA 1: Treinar agressivo nos 3 ambientes (25M timesteps)
Write-Host "ETAPA 1: Treinando 25M timesteps..." -ForegroundColor Cyan
Write-Host "- 5x5 (peso 3)" -ForegroundColor Gray
Write-Host "- 10x10 (peso 4)" -ForegroundColor Gray
Write-Host "- 20x20 (peso 5)" -ForegroundColor Gray
Write-Host ""

& $PY train_grid_world_cpp.py train 10 12 500 25000000 `
  --run-name modelo_90percent `
  --eval-episodes 100 `
  --train-envs "5:3:200:3,10:12:500:4,20:48:1000:5" `
  --view-radius 2 `
  --seed 123

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro no treinamento!" -ForegroundColor Red
    exit 1
}

# Encontrar o modelo treinado
$modelo = Get-ChildItem -Path data -Filter "*modelo_90percent*.zip" | Where-Object { $_.Name -match "curriculum" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($null -eq $modelo) {
    Write-Host "Erro: Modelo nao encontrado!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "TESTANDO NOS 3 AMBIENTES" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "Modelo: $($modelo.Name)" -ForegroundColor Yellow
Write-Host ""

function TestEnvironment($dim, $obs, $steps, $name) {
    Write-Host "Testando $name..." -ForegroundColor Cyan
    $output = & $PY train_grid_world_cpp.py test $dim $obs $steps --model-path "$($modelo.FullName)" --episodes 100 --view-radius 2 2>&1
    $output | Select-String "Full Coverage|Average Coverage"
    $coverage = $output | Select-String "Average Coverage" | ForEach-Object { if ($_ -match "Average Coverage: ([\d.]+)%") { [double]$matches[1] } else { 0 } }
    return $coverage
}

# Testar inicial
$cov5 = TestEnvironment 5 3 200 "5x5"
$cov10 = TestEnvironment 10 12 500 "10x10"
$cov20 = TestEnvironment 20 48 1000 "20x20"

Write-Host ""
Write-Host "Cobertura atual: 5x5=$cov5% | 10x10=$cov10% | 20x20=$cov20%" -ForegroundColor Yellow

# BOOSTS ate atingir 90% em TODOS
$boost_round = 1
$min_coverage = [Math]::Min($cov5, [Math]::Min($cov10, $cov20))

while ($min_coverage -lt $TARGET_COVERAGE -and $boost_round -le 10) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Yellow
    Write-Host "BOOST ROUND $boost_round" -ForegroundColor Yellow
    Write-Host "Min coverage: $min_coverage% (alvo: $TARGET_COVERAGE%)" -ForegroundColor Yellow
    Write-Host "================================" -ForegroundColor Yellow
    Write-Host ""
    
    & $PY train_grid_world_cpp.py curriculum 10 12 500 10000000 `
        --model-path "$($modelo.FullName)" `
        --run-name "modelo_90percent_boost$boost_round" `
        --eval-episodes 100 `
        --train-envs "5:3:200:3,10:12:500:4,20:48:1000:5" `
        --view-radius 2 `
        --seed $((123 + $boost_round * 100))
    
    # Encontrar novo modelo
    $modelo = Get-ChildItem -Path data -Filter "*modelo_90percent_boost$boost_round*.zip" | Where-Object { $_.Name -match "curriculum" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($null -eq $modelo) {
        Write-Host "Erro: Modelo boost$boost_round nao encontrado!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "Testando apos boost $boost_round..." -ForegroundColor Cyan
    
    $cov5 = TestEnvironment 5 3 200 "5x5"
    $cov10 = TestEnvironment 10 12 500 "10x10"
    $cov20 = TestEnvironment 20 48 1000 "20x20"
    
    Write-Host ""
    Write-Host "Resultados: 5x5=$cov5% | 10x10=$cov10% | 20x20=$cov20%" -ForegroundColor Yellow
    
    $min_coverage = [Math]::Min($cov5, [Math]::Min($cov10, $cov20))
    
    if ($min_coverage -ge $TARGET_COVERAGE) {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Green
        Write-Host "SUCESSO! TODOS ACIMA DE 90%!" -ForegroundColor Green
        Write-Host "================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Modelo final: $($modelo.Name)" -ForegroundColor Green
        Write-Host ""
        Write-Host "Resultados finais:" -ForegroundColor Green
        Write-Host "5x5: $cov5%" -ForegroundColor White
        Write-Host "10x10: $cov10%" -ForegroundColor White
        Write-Host "20x20: $cov20%" -ForegroundColor White
        Write-Host ""
        exit 0
    }
    
    $boost_round++
}

Write-Host ""
Write-Host "================================" -ForegroundColor Yellow
Write-Host "COMPLETOU 10 BOOSTS" -ForegroundColor Yellow
Write-Host "Min coverage: $min_coverage%" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Modelo atual:" -ForegroundColor Yellow
Write-Host "5x5: $cov5%" -ForegroundColor White
Write-Host "10x10: $cov10%" -ForegroundColor White
Write-Host "20x20: $cov20%" -ForegroundColor White
Write-Host ""
Write-Host "Execute o script novamente para mais boosts!" -ForegroundColor Yellow
Write-Host ""
