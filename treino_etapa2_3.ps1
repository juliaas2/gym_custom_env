$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"

# Encontrar o melhor modelo da etapa 1
$modelo_5x5 = Get-ChildItem -Path data -Filter "*modelo_final*.zip" | Where-Object { $_.Name -match "modelo_final" -and $_.Name -notmatch "10x10|20x20" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $modelo_5x5) {
    Write-Host "Erro: Modelo 5x5 não encontrado!" -ForegroundColor Red
    exit 1
}
Write-Host "Usando modelo 5x5: $($modelo_5x5.Name)" -ForegroundColor Yellow

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "ETAPA 2: Treinando 10x10" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
& $PY train_grid_world_cpp.py curriculum 10 12 500 1500000 --model-path "$($modelo_5x5.FullName)" --run-name modelo_final_10x10 --eval-episodes 50 --view-radius 2

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro na ETAPA 2! Abortando..." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "ETAPA 3: Treinando 20x20" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# Encontrar o melhor modelo da etapa 2
$modelo_10x10 = Get-ChildItem -Path data -Filter "*modelo_final_10x10*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $modelo_10x10) {
    Write-Host "Erro: Modelo 10x10 não encontrado!" -ForegroundColor Red
    exit 1
}
Write-Host "Carregando modelo: $($modelo_10x10.Name)" -ForegroundColor Yellow
& $PY train_grid_world_cpp.py curriculum 20 48 1000 3000000 --model-path "$($modelo_10x10.FullName)" --run-name modelo_final_20x20 --eval-episodes 50 --view-radius 2

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro na ETAPA 3! Abortando..." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "TREINO COMPLETO! Testando nos 3 ambientes..." -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# Encontrar o melhor modelo da etapa 3
$modelo_final = Get-ChildItem -Path data -Filter "*modelo_final_20x20*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $modelo_final) {
    Write-Host "Erro: Modelo final 20x20 não encontrado!" -ForegroundColor Red
    exit 1
}
Write-Host "Usando modelo final: $($modelo_final.Name)" -ForegroundColor Yellow

Write-Host ""
Write-Host "Testando 5x5..." -ForegroundColor Cyan
& $PY train_grid_world_cpp.py test 5 3 200 --model-path "$($modelo_final.FullName)" --episodes 50 --view-radius 2

Write-Host ""
Write-Host "Testando 10x10..." -ForegroundColor Cyan
& $PY train_grid_world_cpp.py test 10 12 500 --model-path "$($modelo_final.FullName)" --episodes 50 --view-radius 2

Write-Host ""
Write-Host "Testando 20x20..." -ForegroundColor Cyan
& $PY train_grid_world_cpp.py test 20 48 1000 --model-path "$($modelo_final.FullName)" --episodes 50 --view-radius 2

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "✅ TUDO FINALIZADO!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host "Modelo salvo em: $($modelo_final.FullName)" -ForegroundColor Green
