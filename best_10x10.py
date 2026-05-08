import glob
import json
import os

print('MELHORES MODELOS NO 10x10')
print('=' * 80)
print()

resultados = []

# Procurar por summaries de 10x10
files = sorted(glob.glob('data/*10x10*_test_eval_summary.json'), reverse=True)

for summary_path in files[:10]:
    modelo_name = os.path.basename(summary_path).replace('_test_eval_summary.json', '')
    
    with open(summary_path) as f:
        d = json.load(f)
        avg_cov = d.get('average_coverage', 0) * 100
        full_cov = d.get('full_coverage_rate', 0) * 100
        
        resultados.append({
            'modelo': modelo_name,
            'avg': avg_cov,
            'full': full_cov,
            'path': summary_path
        })

# Ordenar por Average Coverage
resultados = sorted(resultados, key=lambda x: x['avg'], reverse=True)

print('TOP 5 MODELOS NO 10x10:')
print()

for i, r in enumerate(resultados[:5], 1):
    status = '✅' if r['avg'] >= 90 else '⚠️' if r['avg'] >= 80 else '❌'
    print(f'{i}. {status} Avg: {r["avg"]:.1f}% | Full: {r["full"]:.1f}%')
    print(f'   {r["modelo"]}')
    print()

print(f'MELHOR: {resultados[0]["modelo"]}')
print(f'Average Coverage: {resultados[0]["avg"]:.1f}%')
