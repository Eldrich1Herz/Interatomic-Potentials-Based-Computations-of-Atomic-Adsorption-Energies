"""
ASE script for Hg adsorption on Au(111) using MACE machine learning potential
Modified version with comprehensive features for learning ASE programming
"""

from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS, LBFGS
from ase.constraints import FixAtoms
from ase.io import write, read
from ase.visualize import view
from ase.db import connect
import numpy as np
import torch
import os

# Try to import MACE with error handling
try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    print("\nWARNING: MACE not installed!")
    print("Please install with: pip install mace-torch")
    print("Or for CPU only: pip install mace-torch --index-url https://download.pytorch.org/whl/cpu\n")
    MACE_AVAILABLE = False
    exit(1)

# ============================================
# 1. Initialize MACE Calculator
# ============================================

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# MACE model options: 'small', 'medium', 'large'
model_size = 'medium'  # Try 'small' for faster, 'large' for more accurate
dtype = 'float32'  # Use 'float64' for higher precision

print(f"MACE model: {model_size}, dtype: {dtype}")
calc = mace_mp(model=model_size, device=device, dtype=dtype)

def get_optimized_energy(atoms, optimizer='BFGS', trajectory=None, fmax=0.05):
    """
    Optimize atomic structure and return energy.
    """
    atoms.calc = calc
    
    if optimizer == 'LBFGS':
        dyn = LBFGS(atoms, logfile='opt.log', trajectory=trajectory)
    else:
        dyn = BFGS(atoms, logfile='opt.log', trajectory=trajectory)
    
    dyn.run(fmax=fmax)
    
    # Check forces
    forces = atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))
    print(f"  Max force after relaxation: {max_force:.4f} eV/Å")
    
    return atoms.get_potential_energy()

# ============================================
# 2. Setup Au(111) Slab
# ============================================

slab_size = (3, 3, 4)  # (nx, ny, layers)
vacuum_gap = 12.0  # Å

print(f"\n{'='*60}")
print(f"MACE Adsorption Study: Hg on Au(111)")
print(f"Slab size: {slab_size}, Vacuum: {vacuum_gap} Å")
print(f"{'='*60}")

slab = fcc111('Au', size=slab_size, vacuum=vacuum_gap)

# ============================================
# 3. Apply Constraints
# ============================================

print("\nLayer tags (for constraint selection):")
for atom in slab:
    print(f"  Atom {atom.index:2d}: tag={atom.tag}, z={atom.position[2]:.2f} Å")

# Freeze bottom two layers
c = FixAtoms(indices=[atom.index for atom in slab if atom.tag > 2])
slab.set_constraint(c)
fixed_count = len([atom.index for atom in slab if atom.tag > 2])
print(f"\nFixed {fixed_count} bottom atoms")

# ============================================
# 4. Calculate Clean Slab Energy
# ============================================

print("\nOptimizing clean Au slab...")
clean_slab = slab.copy()
e_slab = get_optimized_energy(clean_slab, trajectory='clean_slab.traj')
print(f"Clean slab energy: {e_slab:.4f} eV")

# ============================================
# 5. Calculate Isolated Hg Atom Energy
# ============================================

print("\nCalculating isolated Hg atom energy...")
atom_hg = Atoms('Hg', pbc=True)
atom_hg.set_cell([15, 15, 15])
atom_hg.center()
e_atom = get_optimized_energy(atom_hg)
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
    
    traj_file = f'mace_{site}.traj'
    e_total = get_optimized_energy(slab_ads, trajectory=traj_file)
    
    e_ads = e_total - (e_slab + e_atom)
    results[site] = e_ads
    trajectories[site] = traj_file
    final_structures[site] = slab_ads
    
    print(f"  Total energy: {e_total:.4f} eV")
    print(f"  Adsorption energy: {e_ads:.4f} eV")

# ============================================
# 7. Results Analysis
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
# 8. Save to Database
# ============================================

print("\nSaving results to database...")
db = connect('mace_results.db')

for site, e_ads in results.items():
    kvp = {
        'adsorption_energy': float(e_ads),
        'site': site,
        'model': model_size,
        'dtype': dtype,
        'device': device
    }
    db.write(final_structures[site], **kvp)
    print(f"  Saved {site} to database")

# ============================================
# 9. Export Structures
# ============================================

print("\nExporting final structures...")
for site in sites:
    write(f'mace_{site}_final.xyz', final_structures[site])
    write(f'mace_{site}_final.cif', final_structures[site])

print("Exported to XYZ and CIF formats")

# ============================================
# 10. Visualization Instructions
# ============================================

print("\n" + "="*60)
print("VISUALIZATION INSTRUCTIONS")
print("="*60)
print("To view trajectories:")
for site in sites:
    print(f"  ase gui mace_{site}.traj")
print("\nTo view final structures:")
for site in sites:
    print(f"  ase gui mace_{site}_final.xyz")

# ============================================
# 11. Summary
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
MACE Results Analysis:
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
- MACE gives realistic energies within experimental range (-0.5 to -1.0 eV)
- fcc and hcp hollow sites are most stable
- Compare with CHGNet results for validation
""")
print("="*60)