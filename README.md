# semi_sup_cp
Semi-Supervised Sequence Classification through Change Point Detectio

This repository contains code for "Semi-supervised sequence classification through Change Point Detection"


Requirements

- Python 3.7+
- Numpy
- Pytorch
- MATLAB 2018+



The main.py script is used to train neural networks to obtain representations through sim/dissim pairs from change point detection.

The detect_cp folder contains the script run_cp.py which is used to detect change points. The script outputs similar dissimilar pairs which are used by main.py
