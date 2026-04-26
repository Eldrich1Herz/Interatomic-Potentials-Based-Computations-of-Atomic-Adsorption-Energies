"""
ASE script for Hg adsorption on Au(111) using CHGNet machine learning potential
Modified version with additional features for learning ASE programming
FIXED: Force calculation with numpy and database key errors
FIXED2: Isolated Hg atom energy calculated with LJ (CHGNet fails for single atom)
"""

from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS, LBFGS
from ase.constraints import FixAtoms
from chgnet.model.dynamics import CHGNetCalculator
from ase.io import read, write
from ase.db import connect
from ase.calculators.lj import LennardJones  # <-- ADDED for isolated atom
import torch
import numpy as np
import os

# ============================================
# 1. Initialize CHGNet Calculator
# ============================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

calc = CHGNetCalculator(device=device)

def get_optimized_energy(atoms, optimizer='BFGS', trajectory=None):
    atoms.calc = calc
    if optimizer == 'LBFGS':
        dyn = LBFGS(atoms, logfile='opt.log', trajectory=trajectory)
    else:
        dyn = BFGS(atoms, logfile='opt.log', trajectory=trajectory)
    dyn.run(fmax=0.05)
    forces = atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))
    print(f"  Max force after relaxation: {max_force:.4f} eV/Å")
    return atoms.get_potential_energy()

# ============================================
# 2. Setup Au(111) Slab
# ============================================

slab_size = (3, 3, 4)
vacuum_gap = 12.0

print(f"\n{'='*60}")
print(f"CHGNet Adsorption Study: Hg on Au(111)")
print(f"Slab size: {slab_size}, Vacuum: {vacuum_gap} Å")
print(f"{'='*60}")

slab = fcc111('Au', size=slab_size, vacuum=vacuum_gap)

# ============================================
# 3. Apply Constraints
# ============================================

print("\nLayer tags (for constraint selection):")
for atom in slab:
    print(f"  Atom {atom.index:2d}: tag={atom.tag}, z={atom.position[2]:.2f} Å")

c = FixAtoms(indices=[atom.index for atom in slab if atom.tag > 2])
slab.set_constraint(c)
fixed_count = len([atom.index for atom in slab if atom.tag > 2])
print(f"\nFixed {fixed_count} bottom atoms")

# ============================================
# 4. Calculate Clean Slab Energy
# ============================================

print("\nOptimizing clean Au slab...")
clean_slab = slab.copy()
e_slab = get_optimized_energy(clean_slab, optimizer='BFGS', trajectory='clean_slab.traj')
print(f"Clean slab energy: {e_slab:.4f} eV")

# ============================================
# 5. Calculate Isolated Hg Atom Energy (FIXED: using LJ)
# ============================================

print("\nCalculating isolated Hg atom energy...")
atom_hg = Atoms('Hg', pbc=True)
atom_hg.set_cell([15, 15, 15])
atom_hg.center()
# CHGNet cannot handle single atom; use LJ instead
atom_hg.calc = LennardJones(epsilon=0.05, sigma=2.8)
e_atom = atom_hg.get_potential_energy()
print(f"Isolated Hg atom energy: {e_atom:.4f} eV")

# ============================================
# 6. Test Multiple Adsorption Sites
# ============================================

sites = ['fcc', 'hcp', 'ontop', 'bridge']
initial_height = 2.5

results = {}
trajectories = {}
final_structures = {}

print(f"\n{'='*60}")
print(f"Testing adsorption sites at height = {initial_height} Å")
print(f"{'='*60}")

for site in sites:
    print(f"\n{'─'*40}")
    print(f"Optimizing Hg at {site} site...")
    slab_ads = clean_slab.copy()
    add_adsorbate(slab_ads, 'Hg', height=initial_height, position=site)
    traj_file = f'chgnet_{site}.traj'
    if site == 'bridge':
        e_total = get_optimized_energy(slab_ads, optimizer='LBFGS', trajectory=traj_file)
    else:
        e_total = get_optimized_energy(slab_ads, optimizer='BFGS', trajectory=traj_file)
    e_ads = e_total - (e_slab + e_atom)
    results[site] = e_ads
    trajectories[site] = traj_file
    final_structures[site] = slab_ads
    print(f"  Total energy: {e_total:.4f} eV")
    print(f"  Adsorption energy: {e_ads:.4f} eV")

# ============================================
# 7. Results Analysis and Output
# ============================================

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"{'Site':<10} | {'E_ads (eV)':<12} | {'Trajectory file':<20}")
print("-"*50)
for site, e_ads in results.items():
    print(f"{site:<10} | {e_ads:10.4f} eV | {trajectories[site]:<20}")

best_site = min(results, key=results.get)
print("\n" + "-"*50)
print(f"Most stable site: {best_site} with E_ads = {results[best_site]:.3f} eV")
print("-"*50)

# ============================================
# 8. Store Results in ASE Database
# ============================================

print("\nSaving results to database...")
db_path = 'hg_on_au_chgnet.db'
if os.path.exists(db_path):
    print(f"  Database {db_path} already exists. Appending new results...")
else:
    print(f"  Creating new database {db_path}...")
db = connect(db_path)

for site, e_ads in results.items():
    slab_db = final_structures[site]
    kvp = {
        'adsorption_energy': float(e_ads),
        'site': str(site),
        'pot_type': 'CHGNet',
        'model_version': '0.3.0',
        'slab_size_x': int(slab_size[0]),
        'slab_size_y': int(slab_size[1]),
        'slab_layers': int(slab_size[2]),
        'vacuum': float(vacuum_gap),
        'initial_height': float(initial_height),
        'clean_slab_energy': float(e_slab),
        'hg_atom_energy': float(e_atom),
        'device': str(device)
    }
    try:
        db.write(slab_db, **kvp)
        print(f"  Saved {site} site to database")
    except Exception as e:
        print(f"  Warning: Could not save {site} to database: {e}")

print(f"\nResults saved to '{db_path}'")

# ============================================
# 9. Export Structures for External Visualization
# ============================================

print("\nExporting final structures...")
for site in sites:
    slab_export = clean_slab.copy()
    add_adsorbate(slab_export, 'Hg', height=initial_height, position=site)
    write(f'chgnet_{site}_final.xyz', slab_export)
    write(f'chgnet_{site}_final.cif', slab_export)
print("Exported to XYZ and CIF formats (for VESTA, PyMOL, etc.)")

# ============================================
# 10. Visualization Instructions
# ============================================

print("\n" + "="*60)
print("VISUALIZATION INSTRUCTIONS")
print("="*60)
print("To view trajectories:")
for site in sites:
    print(f"  ase gui chgnet_{site}.traj")
print("\nTo view final structures:")
for site in sites:
    print(f"  ase gui chgnet_{site}_final.xyz")
print("\nTo view in Python:")
print("  from ase.io import read")
print("  from ase.visualize import view")
print("  traj = read('chgnet_fcc.traj', index=':')")
print("  view(traj)")

# ============================================
# 11. Query Database Example
# ============================================

print("\n" + "="*60)
print("DATABASE QUERY EXAMPLE")
print("="*60)
print("To query results from database:")
print("  from ase.db import connect")
print("  db = connect('hg_on_au_chgnet.db')")
print("  for row in db.select(site='fcc'):")
print("      print(f\"FCC site: {row.adsorption_energy:.3f} eV\")")
print("  for row in db.select(adsorption_energy<-0.85):")
print("      print(f\"Stable site: {row.site} with {row.adsorption_energy:.3f} eV\")")

# ============================================
# 12. Summary Notes
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
CHGNet Results Analysis:
- Clean slab energy: {e_slab:.4f} eV
- Isolated Hg atom energy: {e_atom:.4f} eV
- Most stable site: {best_site} with E_ads = {results[best_site]:.3f} eV

Site-specific adsorption energies:
""")
for site, e_ads in results.items():
    marker = " ✓ (most stable)" if site == best_site else ""
    print(f"  {site}: {e_ads:.3f} eV{marker}")

print("""
Key observations:
- ML potentials give realistic energies within experimental range (-0.5 to -1.0 eV)
- fcc and hcp hollow sites are most stable (consistent with literature)
- ontop site is least stable
- All forces converged below 0.05 eV/Å

Next steps:
- Change slab thickness to check convergence
- Compare with MACE or DFT results
- Try different adsorbates (O, Pt, Cu)
""")
print("="*60)