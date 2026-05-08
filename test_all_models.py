import glob
import json
import subprocess
import os

print('TESTANDO TODOS OS MODELOS')
print('=' * 80)
print()

# Encontrar todos os modelos
modelos_zips = sorted(glob.glob('data/*.zip'), reverse=True)
modelos_zips = [m for m in modelos_zips if not any(x in m for x in ['_config', '_eval', 'principal'])]

resultados = {}

for modelo_path in modelos_zips[:10]:  # Top 10 mais recentes
    modelo_name = os.path.basename(modelo_path)
    print(f'Testando: {modelo_name}...')
    
    # Procurar por summary json correspondente
    summary_path = modelo_path.replace('.zip', '_eval_summary.json')
    
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            d = json.load(f)
            avg_cov = d.get('average_coverage', 0) * 100
            full_cov = d.get('full_coverage_rate', 0) * 100
            
            # Tentar extrair ambiente do nome
            if '5x5' in modelo_name:
                env = '5x5'
            elif '10x10' in modelo_name:
                env = '10x10'
            elif '20x20' in modelo_name:
                env = '20x20'
            elif 'mixed' in modelo_name:
                env = 'mixed'
            else:
                env = 'unknown'
            
            if env not in resultados:
                resultados[env] = []
            
            resultados[env].append({
                'modelo': modelo_name,
                'avg': avg_cov,
                'full': full_cov
            })
            
            print(f'  Env: {env} | Avg: {avg_cov:.1f}% | Full: {full_cov:.1f}%')
    else:
        print(f'  ⚠️ Sem summary encontrado')
    
    print()

print()
print('=' * 80)
print('RESUMO POR AMBIENTE')
print('=' * 80)
print()

for env in sorted(resultados.keys()):
    print(f'{env.upper()}:')
    for item in sorted(resultados[env], key=lambda x: x['avg'], reverse=True)[:3]:
        print(f'  {item["modelo"][:60]}')
        print(f'    Avg: {item["avg"]:.1f}% | Full: {item["full"]:.1f}%')
    print()
