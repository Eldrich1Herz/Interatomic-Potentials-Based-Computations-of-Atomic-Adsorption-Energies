Automated screening of Hg adsorption sites on Au(111) (Lennard-Jones)
=====================================================================

This script automatically finds all top, bridge, and hollow sites on an
Au(111) slab and relaxes only the adsorbate on each site.

Key features
------------
- Automatic site detection using geometric criteria
- Freezes only the bottom layers (realistic surface model)
- Parallel processing (optional, `--parallel` flag)
- Exports structures to VASP, XYZ, CIF, and PNG
- Saves results to `adsorption_screening.db`

Detected sites
--------------
+-----------+-------+
| Type      | Count |
+===========+=======+
| top       | 9     |
| bridge    | 16    |
| hollow    | 8     |
+-----------+-------+

Most stable: hollow sites with E_ads = –0.8227 eV.
All hollow sites give identical energy, confirming surface symmetry.

Limitation
----------
Lennard-Jones is used here only to demonstrate automation.
For quantitative accuracy, use ML potentials (scripts 2 and 3).

Run
---
::

   # Serial run
   python3 Hg_on_gold_screen_LJ.py

   # Parallel (use all cores)
   python3 Hg_on_gold_screen_LJ.py --parallel

   # Test only first 10 sites
   python3 Hg_on_gold_screen_LJ.py --sites 10

   # Disable visualisation
   python3 Hg_on_gold_screen_LJ.py --no-viz
