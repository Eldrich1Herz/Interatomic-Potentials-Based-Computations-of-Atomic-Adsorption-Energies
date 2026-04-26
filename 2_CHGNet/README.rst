Hg adsorption on Au(111) – CHGNet (machine‑learning potential)
===============================================================

This script uses the pre‑trained CHGNet potential (trained on DFT data).
It automatically tests four adsorption sites and stores results in an ASE database.

Key features
------------
- GPU acceleration if available (auto‑detected)
- Tests fcc, hcp, ontop, bridge sites
- Saves trajectories and final structures
- Stores metadata in `hg_on_au_chgnet.db`

Results
-------
+--------+------------------+
| Site   | E_ads (eV)       |
+========+==================+
| fcc    | –0.895           |
| hcp    | –0.893           |
| ontop  | –0.718           |
| bridge | –0.857           |
+--------+------------------+

**Conclusion:** CHGNet gives adsorption energies within the experimental range
(–0.5 … –1.0 eV). Hollow sites are most stable, consistent with literature.

Requirements
------------
- ASE
- CHGNet (pip install chgnet)
- PyTorch
- NumPy

Run
---
::

   python3 Hg_on_Au-slab_chgnet.py
