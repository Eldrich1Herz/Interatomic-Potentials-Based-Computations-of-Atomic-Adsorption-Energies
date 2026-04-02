Migration from Open Babel to ASE: Hg adsorption on Au(111)
===========================================================

This script compares two approaches:
1. **Original style** using Open Babel with the UFF force field.
2. **Pure ASE** using Lennard-Jones (with proper constraints).

Why this script exists
----------------------
It demonstrates the limitations of classical molecular force fields (UFF)
for periodic metal surfaces and shows how ASE provides a cleaner, more
reliable workflow.

Key observations
----------------
- UFF gives **unphysical** slab energies (~10¹⁴ eV) because it was not
  designed for periodic metallic systems.
- ASE with LJ produces qualitatively reasonable numbers (though still
  too high) and allows proper constraints and trajectory saving.
- This underscores the need for DFT or ML potentials (CHGNet, MACE).

Comparison of results
---------------------
+------------------+------------------------+------------------------+
| Method           | Slab energy (eV)       | E_ads (eV)             |
+==================+========================+========================+
| UFF (Open Babel) | ~ 4.34×10¹⁴            | –4.34×10¹⁴ (unphysical)|
| LJ (ASE)         | –17.83                 | –2.46                  |
| Expected         | ~ –100 (total)         | –0.5 … –1.0            |
+------------------+------------------------+------------------------+

Requirements
------------
- ASE
- Open Babel (optional – script works without it, but the comparison part
  will be skipped)
- NumPy

Run
---
.. code-block:: bash
   python3 5.py
