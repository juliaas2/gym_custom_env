import pandas as pd
import glob
import json

print('MODELO_12H - RESULTADOS POR AMBIENTE')
print('=' * 60)
print()

# Procurar testes mais recentes de cada ambiente
ambientes = [
    ('5x5', '*5x5*_test_eval_summary.json'),
    ('10x10', '*10x10*_test_eval_summary.json'),
    ('20x20', '*20x20*_test_eval_summary.json'),
]

for nome, pattern in ambientes:
    files = sorted(glob.glob(f'data/{pattern}'), reverse=True)
    if files:
        with open(files[0]) as f:
            d = json.load(f)
            avg = d.get('average_coverage', 0) * 100
            full = d.get('full_coverage_rate', 0) * 100
            status = '✅' if avg >= 90 else '❌'
            print(f'{nome}: {status}')
            print(f'  Average Coverage: {avg:.1f}%')
            print(f'  Full Coverage Rate: {full:.1f}%')
            print()

print('META: 90% Average Coverage em TODOS')



