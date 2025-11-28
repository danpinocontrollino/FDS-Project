#!/usr/bin/env python3
"""
Fix for burnout2.ipynb:
1. Create df alias from df_loaded
2. Fix cell 11 that uses df instead of df_loaded
"""
import json
import sys

def fix_notebook(notebook_path):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Find the cell with "df_analysis = df.copy()" and FASE 7
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'df_analysis = df.copy()' in source and 'FASE 7' in source:
                # Replace df.copy() with df_loaded.copy()
                new_source = source.replace('df_analysis = df.copy()', 'df_analysis = df_loaded.copy()')
                nb['cells'][i]['source'] = new_source
                print(f"Fixed cell {i}: replaced df.copy() with df_loaded.copy()")
                break
    
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook saved successfully")

if __name__ == "__main__":
    fix_notebook("notebooks/burnout2.ipynb")
