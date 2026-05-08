$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"
$TARGET_COVERAGE = 90.0

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "TREINO OTIMIZADO - 12 HORAS" -ForegroundColor Green
Write-Host "OBJETIVO: >90% EM TODOS!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "TEMPO ESTIMADO:" -ForegroundColor Yellow
Write-Host "- Treino: 4 horas (12M)" -ForegroundColor Gray
Write-Host "- Boost 1: 4 horas (8M)" -ForegroundColor Gray
Write-Host "- Teste: 2-3 horas" -ForegroundColor Gray
Write-Host "- Boost 2 (opcional): 4 horas" -ForegroundColor Gray
Write-Host ""

# ETAPA 1: Treinar agressivo nos 3 ambientes (12M timesteps)
$inicio = Get-Date
Write-Host "ETAPA 1: Treinando 12M timesteps (FASE 1/2)..." -ForegroundColor Cyan
Write-Host "Inicio: $($inicio.ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

& $PY train_grid_world_cpp.py train 10 12 500 12000000 `
  --run-name modelo_12h `
  --eval-episodes 100 `
  --train-envs "5:3:200:3,10:12:500:4,20:48:1000:5" `
  --view-radius 2 `
  --seed 456

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro no treinamento!" -ForegroundColor Red
    exit 1
}

# Encontrar o modelo treinado
$modelo = Get-ChildItem -Path data -Filter "*modelo_12h*.zip" | Where-Object { $_.Name -notmatch "_config|_eval" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($null -eq $modelo) {
    Write-Host "Erro: Modelo nao encontrado!" -ForegroundColor Red
    exit 1
}

$fim = Get-Date
$duracao_etapa1 = $fim - $inicio
Write-Host ""
Write-Host "Etapa 1 concluida em $($duracao_etapa1.TotalHours.ToString('F1')) horas" -ForegroundColor Green

# TESTE 1
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "TESTE 1" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

function TestEnvironment($dim, $obs, $steps, $name) {
    Write-Host "Testando $name..." -ForegroundColor Gray
    try {
        $output = & $PY train_grid_world_cpp.py test $dim $obs $steps --model-path "$($modelo.FullName)" --episodes 50 --view-radius 2 2>&1 | Where-Object { $_ -notmatch "pkg_resources|Setuptools|UserWarning" }
    } catch {
        Write-Host "Aviso ignorado, continuando..." -ForegroundColor Gray
    }
    $coverage = $output | Select-String "Average Coverage" | ForEach-Object { if ($_ -match "Average Coverage: ([\d.]+)%") { [double]$matches[1] } else { 0 } }
    Write-Host "${name}: ${coverage}%" -ForegroundColor White
    return $coverage
}

$cov5 = TestEnvironment 5 3 200 "5x5"
$cov10 = TestEnvironment 10 12 500 "10x10"
$cov20 = TestEnvironment 20 48 1000 "20x20"

Write-Host ""
Write-Host "Resultados Teste 1: 5x5=$cov5% | 10x10=$cov10% | 20x20=$cov20%" -ForegroundColor Yellow

$min_coverage = [Math]::Min($cov5, [Math]::Min($cov10, $cov20))

# BOOST 1 se necessario
if ($min_coverage -lt $TARGET_COVERAGE) {
    $inicio_boost = Get-Date
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "BOOST 1: 8M timesteps" -ForegroundColor Cyan
    Write-Host "Min coverage: $min_coverage% (alvo: $TARGET_COVERAGE%)" -ForegroundColor Yellow
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "Inicio: $($inicio_boost.ToString('HH:mm:ss'))" -ForegroundColor Gray
    Write-Host ""
    
    & $PY train_grid_world_cpp.py curriculum 10 12 500 8000000 `
        --model-path "$($modelo.FullName)" `
        --run-name "modelo_12h_boost1" `
        --eval-episodes 100 `
        --train-envs "5:3:200:3,10:12:500:4,20:48:1000:5" `
        --view-radius 2 `
        --seed 457
    
    # Encontrar novo modelo
    $modelo = Get-ChildItem -Path data -Filter "*modelo_12h_boost1*.zip" | Where-Object { $_.Name -match "curriculum" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($null -eq $modelo) {
        Write-Host "Erro: Modelo boost1 nao encontrado!" -ForegroundColor Red
        exit 1
    }
    
    $fim_boost = Get-Date
    $duracao_boost = $fim_boost - $inicio_boost
    Write-Host ""
    Write-Host "Boost 1 concluido em $($duracao_boost.TotalHours.ToString('F1')) horas" -ForegroundColor Green
    
    # TESTE 2
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "TESTE 2" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    
    $cov5 = TestEnvironment 5 3 200 "5x5"
    $cov10 = TestEnvironment 10 12 500 "10x10"
    $cov20 = TestEnvironment 20 48 1000 "20x20"
    
    Write-Host ""
    Write-Host "Resultados Teste 2: 5x5=$cov5% | 10x10=$cov10% | 20x20=$cov20%" -ForegroundColor Yellow
    
    $min_coverage = [Math]::Min($cov5, [Math]::Min($cov10, $cov20))
}

Write-Host ""
if ($min_coverage -ge $TARGET_COVERAGE) {
    Write-Host "================================" -ForegroundColor Green
    Write-Host "SUCESSO! TODOS ACIMA DE 90%!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host "Modelo final: $($modelo.Name)" -ForegroundColor Green
    Write-Host ""
    Write-Host "5x5: $cov5%" -ForegroundColor White
    Write-Host "10x10: $cov10%" -ForegroundColor White
    Write-Host "20x20: $cov20%" -ForegroundColor White
} else {
    Write-Host "================================" -ForegroundColor Yellow
    Write-Host "TEMPO DE 12H EXPIROU" -ForegroundColor Yellow
    Write-Host "Min coverage: $min_coverage%" -ForegroundColor Yellow
    Write-Host "================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Modelo atual: $($modelo.Name)" -ForegroundColor Yellow
    Write-Host "5x5: $cov5%" -ForegroundColor White
    Write-Host "10x10: $cov10%" -ForegroundColor White
    Write-Host "20x20: $cov20%" -ForegroundColor White
    Write-Host ""
    Write-Host "Para continuar, rode:" -ForegroundColor Cyan
    Write-Host "python train_grid_world_cpp.py curriculum 10 12 500 8000000 --model-path data/$($modelo.Name) --run-name modelo_12h_boost2 --train-envs `"5:3:200:3,10:12:500:4,20:48:1000:5`" --view-radius 2" -ForegroundColor Gray
}

Write-Host ""
