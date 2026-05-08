$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"
$TARGET_COVERAGE = 90.0

# Encontrar o melhor modelo 20x20
$modelo_atual = Get-ChildItem -Path data -Filter "*modelo_final_20x20*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($null -eq $modelo_atual) {
    Write-Host "Erro: Nenhum modelo 20x20 encontrado!" -ForegroundColor Red
    exit 1
}

Write-Host "Modelo inicial: $($modelo_atual.Name)" -ForegroundColor Yellow
$round = 1

while ($round -le 5) {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "BOOST ROUND ${round}: Treinando mais 5 MILHÕES de timesteps" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Modelo de partida: $($modelo_atual.Name)" -ForegroundColor Yellow
    
    & $PY train_grid_world_cpp.py curriculum 20 48 1000 5000000 `
        --model-path "$($modelo_atual.FullName)" `
        --run-name "modelo_final_boost${round}" `
        --eval-episodes 50 `
        --view-radius 2 `
        --seed $((42 + $round * 100))
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Erro no BOOST ${round}! Abortando..." -ForegroundColor Red
        exit 1
    }
    
    # Encontrar o novo melhor modelo
    $modelo_novo = Get-ChildItem -Path data -Filter "*modelo_final_boost${round}*best.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($null -eq $modelo_novo) {
        Write-Host "Erro: Modelo boost${round} não encontrado!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "Testando boost${round} no 20x20..." -ForegroundColor Cyan
    
    # Testar e capturar cobertura
    $output = & $PY train_grid_world_cpp.py test 20 48 1000 `
        --model-path "$($modelo_novo.FullName)" `
        --episodes 50 `
        --view-radius 2 2>&1
    
    # Extrair cobertura
    $coverage_line = $output | Select-String "Average Coverage"
    if ($coverage_line) {
        Write-Host $coverage_line -ForegroundColor Cyan
        
        # Extrair número de cobertura
        if ($coverage_line -match "Average Coverage: ([\d.]+)%") {
            $coverage = [double]$matches[1]
            Write-Host "Cobertura: $coverage%" -ForegroundColor Yellow
            
            if ($coverage -ge $TARGET_COVERAGE) {
                Write-Host ""
                Write-Host "✅ SUCESSO! Cobertura $coverage% >= $TARGET_COVERAGE% !!!" -ForegroundColor Green
                Write-Host "Modelo final: $($modelo_novo.Name)" -ForegroundColor Green
                
                # Testar nos 3 ambientes
                Write-Host ""
                Write-Host "Testando nos 3 ambientes..." -ForegroundColor Cyan
                Write-Host ""
                
                & $PY train_grid_world_cpp.py test 5 3 200 --model-path "$($modelo_novo.FullName)" --episodes 50 --view-radius 2
                Write-Host ""
                & $PY train_grid_world_cpp.py test 10 12 500 --model-path "$($modelo_novo.FullName)" --episodes 50 --view-radius 2
                Write-Host ""
                & $PY train_grid_world_cpp.py test 20 48 1000 --model-path "$($modelo_novo.FullName)" --episodes 50 --view-radius 2
                
                exit 0
            }
        }
    }
    
    $modelo_atual = $modelo_novo
    $round++
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Yellow
Write-Host "⚠️  Completou 5 rounds mas cobertura ainda < 90%" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow
Write-Host "Último modelo: $($modelo_atual.Name)" -ForegroundColor Yellow
Write-Host "Execute novamente o script para mais 5 rounds!" -ForegroundColor Yellow
