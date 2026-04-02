Hg adsorption on Au(111) – MACE (machine‑learning potential)
=============================================================

This script employs the MACE potential, another state‑of‑the‑art ML model.
It follows the same workflow as the CHGNet script, allowing direct comparison.

Key features
------------
- Three model sizes: `small`, `medium` (default), `large`
- Float32 / float64 precision choice
- Adjustable force convergence criterion (`fmax`)
- Saves trajectories and final structures

Results
-------
+--------+------------------+
| Site   | E_ads (eV)       |
+========+==================+
| fcc    | –0.728           |
| hcp    | –0.725           |
| ontop  | –0.571           |
| bridge | –0.709           |
+--------+------------------+

**Conclusion:** MACE also predicts hollow sites as most stable, with energies
slightly higher than CHGNet (difference ~0.17 eV). Both ML potentials outperform
classical force fields.

Requirements
------------
- ASE
- MACE (pip install mace-torch)
- PyTorch
- NumPy

Run
---
::

   python3 Hg_on_Au-slab_ML.py
