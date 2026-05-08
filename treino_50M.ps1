$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "TREINO NOVO - FOCO FULL COVERAGE" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Estrategia: 50M timesteps mixed" -ForegroundColor Yellow
Write-Host "Seed: 789" -ForegroundColor Yellow
Write-Host "Foco: Maximizar full_coverage_rate" -ForegroundColor Yellow
Write-Host ""

$inicio = Get-Date
Write-Host "Iniciando treino 50M timesteps..." -ForegroundColor Cyan
Write-Host "Inicio: $($inicio.ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

& $PY train_grid_world_cpp.py train 10 12 500 50000000 `
  --run-name modelo_50M_full_coverage `
  --eval-episodes 100 `
  --train-envs "5:3:200:3,10:12:500:4,20:48:1000:5" `
  --view-radius 2 `
  --seed 789

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro no treinamento!" -ForegroundColor Red
    exit 1
}

$fim = Get-Date
$duracao = $fim - $inicio
Write-Host ""
Write-Host "Treino concluido em $($duracao.TotalHours.ToString('F1')) horas" -ForegroundColor Green

# Encontrar modelo
$modelo = Get-ChildItem -Path data -Filter "*modelo_50M_full_coverage*.zip" | Where-Object { $_.Name -notmatch "_config|_eval" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($null -eq $modelo) {
    Write-Host "Erro: Modelo nao encontrado!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Modelo: $($modelo.Name)" -ForegroundColor Green

# Testar
Write-Host ""
Write-Host "================================" -ForegroundColor Yellow
Write-Host "TESTES FINAIS" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow
Write-Host ""

function TestEnvironment($dim, $obs, $steps, $name) {
    Write-Host "Testando $name..." -ForegroundColor Gray
    try {
        $output = & $PY train_grid_world_cpp.py test $dim $obs $steps --model-path "$($modelo.FullName)" --episodes 100 --view-radius 2 2>&1 | Where-Object { $_ -notmatch "pkg_resources|Setuptools|UserWarning" }
    } catch {
        Write-Host "Aviso ignorado..." -ForegroundColor Gray
    }
    
    # Parse Average Coverage (target: 90%)
    $coverage = $output | Select-String "Average Coverage" | ForEach-Object { if ($_ -match "Average Coverage: ([\d.]+)%") { [double]$matches[1] } else { 0 } }
    
    # Parse Full Coverage Rate (info only)
    $fullcov = $output | Select-String "Full Coverage Rate" | ForEach-Object { if ($_ -match "Full Coverage Rate: ([\d.]+)%") { [double]$matches[1] } else { 0 } }
    
    Write-Host "${name}: AvgCov=${coverage}% | FullCov=${fullcov}%" -ForegroundColor White
    return $coverage
}

$cov5 = TestEnvironment 5 3 200 "5x5"
$cov10 = TestEnvironment 10 12 500 "10x10"
$cov20 = TestEnvironment 20 48 1000 "20x20"

Write-Host ""
Write-Host "RESULTADOS FINAIS" -ForegroundColor Yellow
Write-Host "5x5:   ${cov5}% Average Coverage" -ForegroundColor Green
Write-Host "10x10: ${cov10}% Average Coverage" -ForegroundColor Green
Write-Host "20x20: ${cov20}% Average Coverage" -ForegroundColor Green

$meta = 90.0
$sucesso = ($cov5 -ge $meta) -and ($cov10 -ge $meta) -and ($cov20 -ge $meta)

if ($sucesso) {
    Write-Host ""
    Write-Host "✅ META ATINGIDA! Todos >= ${meta}% Average Coverage" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️ Alguns ambientes ainda < ${meta}%" -ForegroundColor Yellow
}

# Salvar como modelo_principal
Write-Host ""
Write-Host "Salvando como modelo_principal.zip..." -ForegroundColor Cyan
Copy-Item $modelo.FullName "data/modelo_principal.zip" -Force
Write-Host "Pronto!" -ForegroundColor Green
