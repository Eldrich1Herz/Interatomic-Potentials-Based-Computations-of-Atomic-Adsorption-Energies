"""
ASE script for Hg adsorption on Au(111) using CHGNet machine learning potential
Modified version with additional features for learning ASE programming
"""

from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS, LBFGS  # Added LBFGS for optimizer switching
from ase.constraints import FixAtoms
from chgnet.model.dynamics import CHGNetCalculator
from ase.io import read, write  # For trajectory analysis and export
from ase.db import connect  # For database storage
import torch  # For GPU detection

# ============================================
# 1. Initialize CHGNet Calculator
# ============================================

# MODIFICATION 11: Run on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# MODIFICATION 15: Specify CHGNet model version
calc = CHGNetCalculator(device=device)  # Add model_name='CHGNet-0.3.0' if needed

def get_optimized_energy(atoms, optimizer='BFGS', trajectory=None):
    """
    Optimize atomic structure and return energy.
    
    Args:
        atoms: ASE Atoms object
        optimizer: 'BFGS', 'LBFGS', or 'FIRE'
        trajectory: filename to save trajectory
    """
    atoms.calc = calc
    
    # MODIFICATION 7: Use different optimization algorithms
    if optimizer == 'LBFGS':
        dyn = LBFGS(atoms, logfile='opt.log', trajectory=trajectory)
    else:  # default BFGS
        dyn = BFGS(atoms, logfile='opt.log', trajectory=trajectory)
    
    # fmax=0.05 is usually sufficient for adsorption energy trends
    dyn.run(fmax=0.05)
    
    # MODIFICATION 14: Check forces after relaxation
    print(f"  Max force after relaxation: {max(atoms.get_forces()):.4f} eV/Å")
    
    return atoms.get_potential_energy()

# ============================================
# 2. Setup Au(111) Slab
# ============================================

# MODIFICATION 2 & 3: Change slab dimensions and vacuum gap
# You can modify these parameters
slab_size = (4, 4, 3)  # (nx, ny, layers) - try (4,4,3) or (2,2,5)
vacuum_gap = 8.0       # Å - try 8.0 or 15.0

print(f"\n{'='*60}")
print(f"CHGNet Adsorption Study: Hg on Au(111)")
print(f"Slab size: {slab_size}, Vacuum: {vacuum_gap} Å")
print(f"{'='*60}")

slab = fcc111('Au', size=slab_size, vacuum=vacuum_gap)

# ============================================
# 3. Apply Constraints
# ============================================

# MODIFICATION 5: Freeze different number of bottom layers
# Print layer tags to understand atom selection
print("\nLayer tags (for constraint selection):")
for atom in slab:
    print(f"  Atom {atom.index}: tag={atom.tag}, z={atom.position[2]:.2f} Å")

# Freeze bottom two layers (original)
# Change to atom.tag == 0 for single layer, or atom.tag > 1 for three layers
c = FixAtoms(indices=[atom.index for atom in slab if atom.tag > 2])
slab.set_constraint(c)
print(f"\nFixed {len([atom.index for atom in slab if atom.tag > 2])} bottom atoms")

# ============================================
# 4. Calculate Clean Slab Energy
# ============================================

print("\nOptimizing clean Au slab...")
# MODIFICATION 9: Save relaxed slab for reuse
clean_slab = slab.copy()  # Store a copy
e_slab = get_optimized_energy(clean_slab, optimizer='BFGS', trajectory='clean_slab.traj')
print(f"Clean slab energy: {e_slab:.4f} eV")

# ============================================
# 5. Calculate Isolated Hg Atom Energy
# ============================================

print("\nCalculating isolated Hg atom energy...")
atom_hg = Atoms('Hg', pbc=True)
atom_hg.set_cell([15, 15, 15])  # Large box for gas phase
atom_hg.center()
e_atom = get_optimized_energy(atom_hg, optimizer='BFGS')
print(f"Isolated Hg atom energy: {e_atom:.4f} eV")

# ============================================
# 6. Test Multiple Adsorption Sites
# ============================================

# MODIFICATION 1: Test additional adsorption sites
sites = ['fcc', 'hcp', 'ontop', 'bridge']  # Added hcp and bridge
# MODIFICATION 4: Adjust initial adsorption height
initial_height = 2.0  # Try 2.0 or 3.0

results = {}
trajectories = {}  # Store trajectory filenames

print(f"\n{'='*60}")
print(f"Testing adsorption sites at height = {initial_height} Å")
print(f"{'='*60}")

# MODIFICATION 9: Reuse the relaxed clean slab
for site in sites:
    print(f"\n{'─'*40}")
    print(f"Optimizing Hg at {site} site...")
    
    # Copy the already relaxed clean slab (more efficient)
    slab_ads = clean_slab.copy()
    
    # Add adsorbate
    add_adsorbate(slab_ads, 'Hg', height=initial_height, position=site)
    
    # MODIFICATION 6: Save trajectory for visualization
    traj_file = f'chgnet_{site}.traj'
    
    # MODIFICATION 7: Try different optimizers (LBFGS for some sites)
    # Use LBFGS for bridge site just as an example
    if site == 'bridge':
        e_total = get_optimized_energy(slab_ads, optimizer='LBFGS', trajectory=traj_file)
    else:
        e_total = get_optimized_energy(slab_ads, optimizer='BFGS', trajectory=traj_file)
    
    e_ads = e_total - (e_slab + e_atom)
    results[site] = e_ads
    trajectories[site] = traj_file
    
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

# MODIFICATION 12: Find most stable site
best_site = min(results, key=results.get)
print("\n" + "-"*50)
print(f"Most stable site: {best_site} with E_ads = {results[best_site]:.3f} eV")
print("-"*50)

# ============================================
# 8. Store Results in ASE Database
# ============================================

# MODIFICATION 13: Use ASE database
db = connect('hg_on_au_chgnet.db')
for site, e_ads in results.items():
    # Create a fresh slab for database storage
    slab_db = fcc111('Au', size=slab_size, vacuum=vacuum_gap)
    add_adsorbate(slab_db, 'Hg', height=initial_height, position=site)
    
    db.write(slab_db, 
             energy=e_ads, 
             site=site,
             calculator='CHGNet',
             slab_size=str(slab_size),
             vacuum=vacuum_gap,
             height=initial_height)

print(f"\nResults saved to 'hg_on_au_chgnet.db'")

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
print("  ase gui chgnet_fcc.traj")
print("  ase gui chgnet_hcp.traj")
print("  ase gui chgnet_ontop.traj")
print("  ase gui chgnet_bridge.traj")
print("\nTo view in Python:")
print("  from ase.io import read")
print("  from ase.visualize import view")
print("  traj = read('chgnet_fcc.traj', index=':')")
print("  view(traj)")

# ============================================
# 11. Summary Notes
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
CHGNet Results Analysis:
- ML potentials give much more realistic energies than LJ
- Values should be closer to experimental range (-0.5 to -1.0 eV)
- Site preference order can be compared with literature
- Trajectories show relaxation path and final geometry

Next steps:
- Change slab thickness to check convergence
- Compare with MACE or DFT results
""")
print("="*60)