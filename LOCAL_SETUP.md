Local setup (no PyTorch)

If you want to run the demo and tests locally without installing heavy binary packages like PyTorch, use the lightweight requirements file and the helper script.

1. Create a virtual environment and install dependencies:

```bash
bash setup_local.sh
```

2. Run streamlit demo (PyTorch not required for interactive UI features):

```bash
source .venv/bin/activate
streamlit run demo_app.py
```

Notes:
- For full model inference (loading `.pt` weights) or training, run on Kaggle or a machine with PyTorch preinstalled (GPU recommended).
- To install full requirements (including PyTorch), use `requirements.txt` on a machine that supports building/wheels or on Kaggle.
