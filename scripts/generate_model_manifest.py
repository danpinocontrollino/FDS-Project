"""Generate a simple manifest JSON for models in models/saved.

Writes `models/saved/manifest.json` containing filename, size, and placeholder metadata.
"""
from pathlib import Path
import json

def main():
    root = Path(__file__).resolve().parent.parent
    model_dir = root / 'models' / 'saved'
    manifest = {}
    if not model_dir.exists():
        print('No models/saved directory found')
        return

    for f in sorted(model_dir.iterdir()):
        if f.is_file():
            manifest[f.name] = {
                'size_bytes': f.stat().st_size,
                'path': str(f.relative_to(root)),
                'notes': ''
            }

    out = model_dir / 'manifest.json'
    with open(out, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print(f'Wrote model manifest to {out}')

if __name__ == '__main__':
    main()
