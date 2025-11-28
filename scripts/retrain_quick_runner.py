"""Runner that executes scripts/retrain_quick.py and appends any exceptions to retrain_quick_log.txt."""
import runpy
import traceback
from pathlib import Path
LOG = Path('results/retrain_quick_log.txt')
try:
    runpy.run_path('scripts/retrain_quick.py', run_name='__main__')
except BaseException as e:
    with open(LOG, 'a') as f:
        f.write('\n--- EXCEPTION (BaseException) ---\n')
        traceback.print_exc(file=f)
    # re-raise so caller sees the same behavior
    raise
else:
    with open(LOG, 'a') as f:
        f.write('\n--- completed without exception ---\n')
