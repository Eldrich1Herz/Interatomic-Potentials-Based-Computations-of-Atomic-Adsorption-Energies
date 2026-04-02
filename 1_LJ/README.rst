Hg adsorption on Au(111) – Lennard-Jones potential
===================================================

This script demonstrates the simplest possible ASE workflow:
building an Au(111) slab, adding a Hg atom, relaxing the structure,
and computing the adsorption energy.

Key features
------------
- Uses Lennard-Jones potential with **explicit parameters** (epsilon=0.05 eV, sigma=2.8 Å)
- Tests four high‑symmetry sites: ontop, fcc, hcp, bridge
- Saves relaxation trajectories (`.traj`) and final structures (`.xyz`, `.cif`)
- Interactive visualisation with `ase.visualize.view()`

Results
-------
+--------+------------------+
| Site   | E_ads (eV)       |
+========+==================+
| ontop  | +2.49 (unstable) |
| fcc    | –0.09            |
| hcp    | –0.09            |
| bridge | +0.05            |
+--------+------------------+

**Conclusion:** Hollow sites (fcc/hcp) are slightly stable, but the binding
energy is an order of magnitude too low – LJ cannot describe the covalent
character of the Hg–Au bond.

Requirements
------------
- ASE ≥ 3.23.0
- NumPy

Run
---
.. code-block:: bash
   python3 Hg_on_Au-slab_LJ.py