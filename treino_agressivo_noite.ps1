$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"
$TARGET_COVERAGE = 90.0

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "TREINO AGRESSIVO - NOITE TODA!" -ForegroundColor Green
Write-Host "10 ROUNDS X 10M TIMESTEPS CADA" -ForegroundColor Green
Write-Host "= 100 MILHOES DE TIMESTEPS!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# Encontrar o melhor modelo 20x20
$modelo_atual = Get-ChildItem -Path data -Filter "*modelo_final_20x20*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($null -eq $modelo_atual) {
    Write-Host "Erro: Nenhum modelo 20x20 encontrado!" -ForegroundColor Red
    exit 1
}

Write-Host "Modelo inicial: $($modelo_atual.Name)" -ForegroundColor Yellow
Write-Host ""
$round = 1

while ($round -le 10) {
    $inicio = Get-Date
    
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "BOOST ROUND $round/10: 10M timesteps" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "Inicio: $($inicio.ToString('HH:mm:ss'))" -ForegroundColor Gray
    Write-Host ""
    
    & $PY train_grid_world_cpp.py curriculum 20 48 1000 10000000 `
        --model-path "$($modelo_atual.FullName)" `
        --run-name "modelo_agressivo_boost$round" `
        --eval-episodes 100 `
        --view-radius 2 `
        --seed $((42 + $round * 200))
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Erro no BOOST $round!" -ForegroundColor Red
        exit 1
    }
    
    # Encontrar o novo melhor modelo (procura por "curriculum" em vez de "best")
    $modelo_novo = Get-ChildItem -Path data -Filter "*modelo_agressivo_boost$round*.zip" | Where-Object { $_.Name -match "curriculum" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($null -eq $modelo_novo) {
        Write-Host "Erro: Modelo boost$round nao encontrado!" -ForegroundColor Red
        Write-Host "Arquivos disponiveis:" -ForegroundColor Yellow
        Get-ChildItem -Path data -Filter "*modelo_agressivo_boost$round*" -Name
        exit 1
    }
    
    Write-Host ""
    Write-Host "Testando modelo boost$round no 20x20..." -ForegroundColor Cyan
    
    # Testar e capturar cobertura
    $output = & $PY train_grid_world_cpp.py test 20 48 1000 `
        --model-path "$($modelo_novo.FullName)" `
        --episodes 100 `
        --view-radius 2 2>&1
    
    # Extrair cobertura
    $coverage_line = $output | Select-String "Average Coverage"
    if ($coverage_line) {
        Write-Host $coverage_line -ForegroundColor Cyan
        
        # Extrair número de cobertura
        if ($coverage_line -match "Average Coverage: ([\d.]+)%") {
            $coverage = [double]$matches[1]
            $full_coverage_line = $output | Select-String "Full Coverage Rate"
            
            Write-Host $full_coverage_line -ForegroundColor Cyan
            Write-Host "Cobertura media: $coverage%" -ForegroundColor Yellow
            
            $fim = Get-Date
            $duracao = $fim - $inicio
            Write-Host "Duracao do round: $($duracao.TotalHours.ToString('F1')) horas" -ForegroundColor Gray
            
            if ($coverage -ge $TARGET_COVERAGE) {
                Write-Host ""
                Write-Host "================================" -ForegroundColor Green
                Write-Host "SUCESSO! COBERTURA $coverage%!" -ForegroundColor Green
                Write-Host "================================" -ForegroundColor Green
                Write-Host "Modelo final encontrado!" -ForegroundColor Green
                Write-Host "Arquivo: $($modelo_novo.Name)" -ForegroundColor Green
                Write-Host ""
                
                # Testar nos 3 ambientes
                Write-Host "Testando modelo final nos 3 ambientes..." -ForegroundColor Cyan
                Write-Host ""
                
                Write-Host "Testando 5x5..." -ForegroundColor Yellow
                & $PY train_grid_world_cpp.py test 5 3 200 --model-path "$($modelo_novo.FullName)" --episodes 100 --view-radius 2
                Write-Host ""
                
                Write-Host "Testando 10x10..." -ForegroundColor Yellow
                & $PY train_grid_world_cpp.py test 10 12 500 --model-path "$($modelo_novo.FullName)" --episodes 100 --view-radius 2
                Write-Host ""
                
                Write-Host "Testando 20x20..." -ForegroundColor Yellow
                & $PY train_grid_world_cpp.py test 20 48 1000 --model-path "$($modelo_novo.FullName)" --episodes 100 --view-radius 2
                
                Write-Host ""
                Write-Host "================================" -ForegroundColor Green
                Write-Host "TREINO CONCLUIDO COM SUCESSO!" -ForegroundColor Green
                Write-Host "================================" -ForegroundColor Green
                Write-Host ""
                exit 0
            }
        }
    }
    
    $modelo_atual = $modelo_novo
    $round++
}

Write-Host ""
Write-Host "================================" -ForegroundColor Yellow
Write-Host "COMPLETOU 10 ROUNDS" -ForegroundColor Yellow
Write-Host "(100 MILHOES DE TIMESTEPS!)" -ForegroundColor Yellow
Write-Host "Cobertura ainda menor que 90%" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ultimo modelo: $($modelo_atual.Name)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Opcoes:" -ForegroundColor Cyan
Write-Host "1. Rode o script novamente para mais 10 rounds" -ForegroundColor Gray
Write-Host "2. Use o modelo atual mesmo com menos de 90%" -ForegroundColor Gray
Write-Host ""
