$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"
$TARGET_COVERAGE = 90.0

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "TESTE E BOOST - CONTINUA" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# Encontrar o modelo treinado (modelo_12h)
$modelo = Get-ChildItem -Path data -Filter "*modelo_12h*.zip" | Where-Object { $_.Name -notmatch "_config|_eval" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($null -eq $modelo) {
    Write-Host "Erro: Modelo modelo_12h nao encontrado!" -ForegroundColor Red
    Write-Host "Execute primeiro: .\treino_12horas.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Modelo encontrado: $($modelo.Name)" -ForegroundColor Green
Write-Host ""

# TESTE 1
Write-Host "================================" -ForegroundColor Yellow
Write-Host "TESTE 1 - Avaliacao inicial" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow
Write-Host ""

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
Write-Host "Resultados Teste 1: 5x5=${cov5}% | 10x10=${cov10}% | 20x20=${cov20}%" -ForegroundColor Yellow

$sucesso_teste1 = ($cov5 -ge $TARGET_COVERAGE) -and ($cov10 -ge $TARGET_COVERAGE) -and ($cov20 -ge $TARGET_COVERAGE)

if ($sucesso_teste1) {
    Write-Host ""
    Write-Host "✅ SUCESSO! Todos os ambientes >= $TARGET_COVERAGE%" -ForegroundColor Green
    Write-Host ""
    Write-Host "Modelo final: $($modelo.Name)" -ForegroundColor Green
    
    # Salvar como modelo_principal
    Write-Host ""
    Write-Host "Salvando como modelo_principal.zip..." -ForegroundColor Cyan
    Copy-Item $modelo.FullName "data/modelo_principal.zip" -Force
    Write-Host "Pronto!" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "Algum ambiente com coverage < $TARGET_COVERAGE%. Executando BOOST..." -ForegroundColor Yellow
Write-Host ""

# BOOST 1
$inicio_boost = Get-Date
Write-Host "ETAPA 2: BOOST (8M timesteps)..." -ForegroundColor Cyan
Write-Host "Carregando modelo: $($modelo.FullName)" -ForegroundColor Gray
Write-Host ""

& $PY train_grid_world_cpp.py curriculum "$($modelo.FullName)" 5 3 200 8000000 `
  --run-name modelo_12h_boost1 `
  --eval-episodes 100 `
  --train-envs "5:3:200:3,10:12:500:4,20:48:1000:5" `
  --view-radius 2 `
  --seed 457

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro no boost!" -ForegroundColor Red
    exit 1
}

# Encontrar modelo com boost
$modelo_boost = Get-ChildItem -Path data -Filter "*modelo_12h_boost1*.zip" | Where-Object { $_.Name -notmatch "_config|_eval" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($null -eq $modelo_boost) {
    Write-Host "Erro: Modelo boost nao encontrado!" -ForegroundColor Red
    exit 1
}

$fim_boost = Get-Date
$duracao_boost = $fim_boost - $inicio_boost
Write-Host ""
Write-Host "Boost concluido em $($duracao_boost.TotalHours.ToString('F1')) horas" -ForegroundColor Green

# TESTE 2
Write-Host ""
Write-Host "================================" -ForegroundColor Yellow
Write-Host "TESTE 2 - Apos boost" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow
Write-Host ""

$cov5_boost = TestEnvironment 5 3 200 "5x5"
$cov10_boost = TestEnvironment 10 12 500 "10x10"
$cov20_boost = TestEnvironment 20 48 1000 "20x20"

Write-Host ""
Write-Host "Resultados Teste 2: 5x5=${cov5_boost}% | 10x10=${cov10_boost}% | 20x20=${cov20_boost}%" -ForegroundColor Yellow

$sucesso_teste2 = ($cov5_boost -ge $TARGET_COVERAGE) -and ($cov10_boost -ge $TARGET_COVERAGE) -and ($cov20_boost -ge $TARGET_COVERAGE)

if ($sucesso_teste2) {
    Write-Host ""
    Write-Host "✅ SUCESSO APOS BOOST! Todos >= $TARGET_COVERAGE%" -ForegroundColor Green
    Write-Host ""
    Write-Host "Modelo final: $($modelo_boost.Name)" -ForegroundColor Green
    
    # Salvar como modelo_principal
    Write-Host ""
    Write-Host "Salvando como modelo_principal.zip..." -ForegroundColor Cyan
    Copy-Item $modelo_boost.FullName "data/modelo_principal.zip" -Force
    Write-Host "Pronto!" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "⚠️ Ainda abaixo de $TARGET_COVERAGE% em alguns ambientes" -ForegroundColor Yellow
Write-Host "Modelos salvos:" -ForegroundColor Yellow
Write-Host "- Original: $($modelo.Name) (5x5=${cov5}% | 10x10=${cov10}% | 20x20=${cov20}%)" -ForegroundColor Gray
Write-Host "- Boost:    $($modelo_boost.Name) (5x5=${cov5_boost}% | 10x10=${cov10_boost}% | 20x20=${cov20_boost}%)" -ForegroundColor Gray
Write-Host ""
Write-Host "Salvando melhor modelo como modelo_principal.zip..." -ForegroundColor Cyan

# Comparar e salvar melhor
$melhor = if (($cov5_boost + $cov10_boost + $cov20_boost) -ge ($cov5 + $cov10 + $cov20)) { $modelo_boost } else { $modelo }
Copy-Item $melhor.FullName "data/modelo_principal.zip" -Force
Write-Host "Melhor modelo: $($melhor.Name)" -ForegroundColor Green
