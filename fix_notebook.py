#!/usr/bin/env python3
import json

new_code = '''# FASE 7: ANALISI DURATA OTTIMALE DELLA PAUSA
# ==================================================
print('=' * 80)
print('FASE 7: ANALISI DURATA OTTIMALE DELLA PAUSA')
print('=' * 80)

# Usa df_loaded che e disponibile nel kernel
df_analysis = df_loaded.copy()

# Verifica le colonne disponibili
print(f'Colonne disponibili: {list(df_analysis.columns)[:10]}...')

# Identificazione dinamica delle colonne chiave
stress_col = 'Stress Level' if 'Stress Level' in df_analysis.columns else 'stress'
sleep_col = 'Sleep Duration' if 'Sleep Duration' in df_analysis.columns else 'sleep'
mood_col = 'Mood' if 'Mood' in df_analysis.columns else 'mood'
hr_col = 'Heart Rate' if 'Heart Rate' in df_analysis.columns else 'heart_rate'

print(f'   Stress column: {stress_col}')
print(f'   Sleep column: {sleep_col}')
print(f'   Mood column: {mood_col}')
print(f'   Heart rate column: {hr_col}')

# Normalizzazione delle colonne per lanalisi (0-1)
for col in [stress_col, sleep_col, mood_col, hr_col]:
    if col in df_analysis.columns:
        col_min = df_analysis[col].min()
        col_max = df_analysis[col].max()
        if col_max > col_min:
            df_analysis[f'{col}_norm'] = (df_analysis[col] - col_min) / (col_max - col_min)
        else:
            df_analysis[f'{col}_norm'] = 0.5

def calculate_recommended_break_duration(stress_level, sleep_duration, heart_rate, mood):
    severity_score = (stress_level * 0.4) + ((1 - sleep_duration) * 0.3) + ((1 - mood) * 0.2) + (heart_rate * 0.1)
    if severity_score < 0.3:
        break_hours = 0.5
    elif severity_score < 0.4:
        break_hours = 1.0
    elif severity_score < 0.5:
        break_hours = 2.0
    elif severity_score < 0.6:
        break_hours = 4.0
    elif severity_score < 0.7:
        break_hours = 8.0
    else:
        break_hours = 24.0
    return break_hours, severity_score

# Calcola durata pausa consigliata
stress_norm = f'{stress_col}_norm'
sleep_norm = f'{sleep_col}_norm'
mood_norm = f'{mood_col}_norm'
hr_norm = f'{hr_col}_norm'

if all(c in df_analysis.columns for c in [stress_norm, sleep_norm, mood_norm, hr_norm]):
    df_analysis['Recommended_Break_Hours'] = df_analysis.apply(
        lambda row: calculate_recommended_break_duration(
            row[stress_norm], row[sleep_norm], row[hr_norm], row[mood_norm]
        )[0], axis=1
    )
    df_analysis['Break_Severity_Score'] = df_analysis.apply(
        lambda row: calculate_recommended_break_duration(
            row[stress_norm], row[sleep_norm], row[hr_norm], row[mood_norm]
        )[1], axis=1
    )
    
    # Statistiche
    break_stats = df_analysis['Recommended_Break_Hours'].value_counts().sort_index()
    print(f'DISTRIBUZIONE DURATA PAUSA CONSIGLIATA:')
    for duration, count in break_stats.items():
        pct = 100*count/len(df_analysis)
        print(f'   {duration:0.1f} ore: {count} record ({pct:.1f}%)')
    
    # Salva analisi
    import os
    os.makedirs('results', exist_ok=True)
    cols_to_save = ['Recommended_Break_Hours', 'Break_Severity_Score', stress_col, sleep_col, mood_col]
    df_analysis[[c for c in cols_to_save if c in df_analysis.columns]].to_csv(
        'results/break_duration_analysis.csv', index=False
    )
    print(f'Analisi pausa salvata in results/break_duration_analysis.csv')
else:
    print(f'Alcune colonne normalizzate non disponibili.')
'''

with open('notebooks/burnout2.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'df_analysis = df.copy()' in source and 'FASE 7' in source:
            lines = new_code.split('\n')
            nb['cells'][i]['source'] = [line + '\n' if j < len(lines) - 1 else line for j, line in enumerate(lines)]
            print(f'Updated cell at index {i}')
            break

with open('notebooks/burnout2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('Done')
