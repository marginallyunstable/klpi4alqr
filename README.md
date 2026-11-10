# KL-Regularized Policy Iteration for Adaptive LQR: Gain Updates and Probing from One Policy Improvement: simulations

This repository holds the simulations of Section VI of

> [1] A. Bhole, M. Mahmoudi Filabadi, G. Crevecoeur, T. Lefebvre, "KL-Regularized Policy Iteration for Adaptive LQR: Gain Updates and Probing from One Policy Improvement", 2026.

The extended version of [1], with all proofs, is [klpi4alqr_extended.pdf](klpi4alqr_extended.pdf); the arXiv link will follow.

```
paper.ipynb    the setup, one run of the four variants as a plain loop, the driver of all runs,
               and Figs. 1 and 2 of [1]
klpi.py        the simulation code
figures/       the LaTeX sources of Figs. 1 and 2
data/          the forty random plants and the optimal costs used for the probing budgets
results/       the per-run results of the paper's runs and the tables the figures read
```

## Use

The code needs Python 3.10 or later with numpy and scipy, JupyterLab, and a TeX Live installation with pgfplots and `pdftoppm` for the figures.

```bash
pip install -r requirements.txt
jupyter lab paper.ipynb
```

To rerun the simulations from a terminal instead:

```bash
python klpi.py --quick    # 24 seeds per plant, about 8 min on 20 cores, into results/quick/
python klpi.py            # the paper's 60960 runs, about 2 h 10 min on 20 cores, into results/
```
