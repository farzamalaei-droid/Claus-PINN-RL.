# Claus-PINN-RL
**Physics-Informed Reinforcement Learning for Optimized Claus Sulfur Recovery and CO₂ Capture**  
Published in AIChE Journal (2025) | DOI: pending

This repository contains all code, trained models, and notebooks to fully reproduce the results reported in the paper:
- Sulfur recovery >99.5 %
- Energy savings 22.5 % (20.0 → 15.5 MW)
- Tail-gas SO₂ ≈ 50 mg/Nm³
- 85 % CO₂ capture with ionic liquid [bmim][Tf₂N]

All random seeds = 42 → 100 % reproducible

### Quick Start
```bash
conda create -n claus python=3.9 -y
conda activate claus
pip install -r requirements.txt

# Train PINN
jupyter notebook 2_PINN/train_pinn.ipynb

# Train RL agent (or use pre-trained)
python 3_RL_Optimization/train_rl.py

# Reproduce all figures
jupyter notebook notebooks/
