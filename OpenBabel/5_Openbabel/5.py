#!/usr/bin/env python
"""
ASE script: Transitioning from Open Babel to ASE for Hg Adsorption on Au(111)
Complete migration example with all ASE-native features
"""

import os
import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS, LBFGS
from ase.constraints import FixAtoms
from ase.io import write, read
from ase.visualize import view
from ase.db import connect
from ase.eos import EquationOfState
from multiprocessing import Pool
import argparse
import time

# ==============================================
# Create output directories
# ==============================================
os.makedirs('outputs', exist_ok=True)
os.makedirs('trajectories', exist_ok=True)
os.makedirs('structures', exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('eos_data', exist_ok=True)

# ==============================================
# Configuration Parameters
# ==============================================

# System parameters
adsorbate = 'Hg'
substrate = 'Au'
vacuum = 10.0  # Å
slab_size = (3, 3, 4)  # (nx, ny, layers)
lattice_constant = 4.08  # Å for Au

# Adsorption parameters
initial_height = 2.5  # Å
sites_to_test = ['fcc', 'hcp', 'ontop', 'bridge']

# Calculator choice
# Options: 'EMT', 'LJ', 'CHGNet', 'MACE'
calculator_type = 'EMT'  # Start with EMT, then try others

# Optimization parameters
optimizer_type = 'BFGS'  # 'BFGS' or 'LBFGS'
fmax_convergence = 0.05  # eV/Å

# Visualization
enable_visualization = True
save_images = True

# Parallelization
use_parallel = False  # Set to True for multiple sites
n_cores = 4

# ==============================================
# Helper Functions
# ==============================================

def setup_calculator(calc_type=calculator_type):
    """
    Set up the appropriate ASE calculator.
    """
    if calc_type == 'EMT':
        return EMT()
    elif calc_type == 'LJ':
        return LennardJones(epsilon=0.05, sigma=2.8, rc=10.0)
    elif calc_type == 'CHGNet':
        try:
            from chgnet.model.dynamics import CHGNetCalculator
            return CHGNetCalculator()
        except ImportError:
            print("CHGNet not installed, falling back to EMT")
            return EMT()
    elif calc_type == 'MACE':
        try:
            from mace.calculators import mace_mp
            return mace_mp(model="medium", device='cpu')
        except ImportError:
            print("MACE not installed, falling back to EMT")
            return EMT()
    else:
        raise ValueError(f"Unknown calculator type: {calc_type}")

def visualize_atoms(atoms, title, filename=None):
    """
    Visualize atoms and optionally save image.
    """
    if enable_visualization:
        print(f"  Visualizing: {title}")
        view(atoms)
    
    if save_images and filename:
        timestamp = time.strftime("%H%M%S")
        full_path = f'structures/{filename}_{timestamp}.png'
        write(full_path, atoms, rotation='90x,90y')
        print(f"  Saved image: {full_path}")

def get_optimized_energy(atoms, calc, traj_file=None, optimizer=optimizer_type):
    """
    Optimize atomic structure and return energy.
    """
    atoms.calc = calc
    
    if optimizer == 'LBFGS':
        dyn = LBFGS(atoms, trajectory=traj_file, logfile='outputs/optimization.log')
    else:
        dyn = BFGS(atoms, trajectory=traj_file, logfile='outputs/optimization.log')
    
    dyn.run(fmax=fmax_convergence)
    
    # Check forces
    max_force = max(abs(atoms.get_forces().flatten()))
    print(f"  Max force after relaxation: {max_force:.4f} eV/Å")
    
    return atoms.get_potential_energy()

# ==============================================
# 1. Build the Slab with ASE (instead of reading from file)
# ==============================================

print("\n" + "="*70)
print("ASE MIGRATION DEMO: Hg on Au(111)")
print("="*70)

print("\n1. Building Au(111) slab with ASE...")
slab = fcc111(substrate, size=slab_size, vacuum=vacuum, a=lattice_constant)
print(f"   Created slab with {len(slab)} atoms")
print(f"   Cell: {slab.cell}")
print(f"   Vacuum: {vacuum} Å")

# Visualize clean slab
visualize_atoms(slab, "Clean Au(111) slab", "clean_slab")

# ==============================================
# 2. Apply Constraints (fix bottom layers)
# ==============================================

print("\n2. Applying constraints to bottom layers...")
z_coords = slab.get_positions()[:, 2]
bottom_threshold = np.min(z_coords) + 2.0  # Adjust based on layer thickness
bottom_indices = [i for i, z in enumerate(z_coords) if z <= bottom_threshold]
constraint = FixAtoms(indices=bottom_indices)
slab.set_constraint(constraint)
print(f"   Fixed {len(bottom_indices)} bottom atoms (z ≤ {bottom_threshold:.2f} Å)")

# ==============================================
# 3. Set up Calculator (instead of Open Babel force field)
# ==============================================

print(f"\n3. Setting up {calculator_type} calculator...")
calc = setup_calculator(calculator_type)
print(f"   Calculator: {calculator_type}")

# ==============================================
# 4. Calculate Clean Slab Energy
# ==============================================

print("\n4. Calculating clean slab energy...")
clean_slab = slab.copy()
traj_file = 'trajectories/clean_slab_relax.traj'
e_slab = get_optimized_energy(clean_slab, calc, traj_file)
print(f"   Clean slab energy: {e_slab:.6f} eV")

# Visualize relaxed slab
visualize_atoms(clean_slab, "Relaxed Au slab", "relaxed_slab")

# ==============================================
# 5. Calculate Isolated Atom Energy
# ==============================================

print("\n5. Calculating isolated Hg atom energy...")
atom = Atoms(adsorbate, pbc=True)
atom.set_cell([20, 20, 20])  # Large box to simulate vacuum
atom.center()
visualize_atoms(atom, "Isolated Hg atom", "isolated_hg")

e_atom = get_optimized_energy(atom, calc)
print(f"   Isolated Hg atom energy: {e_atom:.6f} eV")

# ==============================================
# 6. Test Multiple Adsorption Sites
# ==============================================

print("\n6. Testing multiple adsorption sites...")
print(f"   Sites to test: {sites_to_test}")
print(f"   Initial height: {initial_height} Å")

def run_site_calculation(site):
    """
    Run calculation for a single site.
    """
    print(f"\n   {'─'*40}")
    print(f"   Processing {site} site...")
    
    # Create fresh slab
    slab_site = fcc111(substrate, size=slab_size, vacuum=vacuum, a=lattice_constant)
    
    # Add constraints
    z_coords = slab_site.get_positions()[:, 2]
    bottom_indices = [i for i, z in enumerate(z_coords) if z <= bottom_threshold]
    slab_site.set_constraint(FixAtoms(indices=bottom_indices))
    
    # Add adsorbate
    add_adsorbate(slab_site, adsorbate, height=initial_height, position=site)
    
    # Visualize initial configuration
    if enable_visualization:
        visualize_atoms(slab_site, f"{site} initial", f"{site}_initial")
    
    # Set up calculator
    site_calc = setup_calculator(calculator_type)
    
    # Run optimization
    traj_file = f'trajectories/{site}_relax.traj'
    e_total = get_optimized_energy(slab_site, site_calc, traj_file)
    
    # Calculate adsorption energy
    e_ads = e_total - (e_slab + e_atom)
    
    # Visualize final configuration
    if enable_visualization:
        visualize_atoms(slab_site, f"{site} final", f"{site}_final")
    
    # Save final structure
    write(f'structures/{site}_final.xyz', slab_site)
    write(f'structures/{site}_final.cif', slab_site)
    
    # Get final height
    ads_index = len(slab_site) - 1
    final_height = slab_site.get_positions()[ads_index][2] - np.max(slab.get_positions()[:,2])
    
    print(f"   {site.upper()} site:")
    print(f"     Total energy: {e_total:.6f} eV")
    print(f"     Adsorption energy: {e_ads:.6f} eV")
    print(f"     Final height: {final_height:.3f} Å")
    
    return site, e_ads, e_total, slab_site, final_height

# Run calculations (serial or parallel)
if use_parallel:
    print(f"\n   Running in parallel with {n_cores} cores...")
    with Pool(n_cores) as pool:
        results_list = pool.map(run_site_calculation, sites_to_test)
else:
    print("\n   Running in serial mode...")
    results_list = [run_site_calculation(site) for site in sites_to_test]

# Process results
results = {}
atoms_dict = {}
final_heights = {}
for site, e_ads, e_total, slab_site, height in results_list:
    results[site] = e_ads
    atoms_dict[site] = slab_site
    final_heights[site] = height

# ==============================================
# 7. Display Results
# ==============================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\nCalculator: {calculator_type}")
print(f"Slab energy: {e_slab:.6f} eV")
print(f"Hg atom energy: {e_atom:.6f} eV")
print("\nAdsorption energies:")
print("-"*50)
print(f"{'Site':10s} | {'E_ads (eV)':12s} | {'Final Height (Å)':15s}")
print("-"*50)

for site in sites_to_test:
    print(f"{site:10s} | {results[site]:12.6f} | {final_heights[site]:15.3f}")

# Find most stable site
best_site = min(results, key=results.get)
print("\n" + "-"*50)
print(f"✓ Most stable site: {best_site} with E_ads = {results[best_site]:.6f} eV")
print("-"*50)

# ==============================================
# 8. Store Results in ASE Database
# ==============================================

print("\n8. Saving results to ASE database...")
db_path = 'database/hg_on_au.db'
db = connect(db_path)

for site in sites_to_test:
    kvp = {
        'adsorbate': adsorbate,
        'substrate': substrate,
        'site': site,
        'calculator': calculator_type,
        'adsorption_energy': float(results[site]),
        'final_height': float(final_heights[site]),
        'slab_energy': float(e_slab),
        'atom_energy': float(e_atom)
    }
    
    db.write(atoms_dict[site], **kvp)
    print(f"   Saved {site} site to database")

print(f"\n   Database saved to: {db_path}")

# ==============================================
# 9. Export All Structures
# ==============================================

print("\n9. Exporting structures in multiple formats...")
for site in sites_to_test:
    base = f'structures/{site}_final'
    write(f'{base}.xyz', atoms_dict[site])
    write(f'{base}.cif', atoms_dict[site])
    write(f'{base}.vasp', atoms_dict[site], format='vasp', direct=True)
    print(f"   Exported {site}: .xyz, .cif, .vasp")

# ==============================================
# 10. Bonus: Equation of State for Bulk Au
# ==============================================

print("\n" + "="*70)
print("BONUS: Equation of State for Bulk Au")
print("="*70)

print("\n10. Computing EOS for bulk gold...")
from ase.build import bulk

# Create bulk gold
bulk_au = bulk('Au', 'fcc', a=lattice_constant, cubic=True)

# Calculate EOS
volumes = []
energies = []
scales = np.linspace(0.95, 1.05, 11)

print("\n   Calculating energies at different volumes:")
for scale in scales:
    bulk_scaled = bulk_au.copy()
    bulk_scaled.set_cell(bulk_au.cell * scale, scale_atoms=True)
    bulk_scaled.calc = setup_calculator(calculator_type)
    
    energy = bulk_scaled.get_potential_energy()
    volume = bulk_scaled.get_volume()
    
    volumes.append(volume)
    energies.append(energy)
    print(f"     scale={scale:.3f}, volume={volume:.2f} Å³, energy={energy:.6f} eV")

# Fit EOS
print("\n   Fitting equation of state...")
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()

print(f"\n   Results:")
print(f"     Equilibrium volume: {v0:.2f} Å³")
print(f"     Equilibrium energy: {e0:.6f} eV")
print(f"     Bulk modulus: {B:.2f} GPa")

# Save EOS data
eos_data = np.column_stack([scales, volumes, energies])
np.savetxt('eos_data/bulk_au_eos.txt', eos_data, 
           header='scale volume(A^3) energy(eV)', fmt='%.6f')

# Plot EOS (optional)
try:
    import matplotlib.pyplot as plt
    eos.plot('eos_data/eos_fit.pdf')
    print("   EOS plot saved to: eos_data/eos_fit.pdf")
except ImportError:
    print("   matplotlib not installed, skipping plot")

# ==============================================
# 11. Visualization Instructions
# ==============================================

print("\n" + "="*70)
print("VISUALIZATION INSTRUCTIONS")
print("="*70)
print("\nTo view trajectories:")
for site in sites_to_test:
    print(f"  ase gui trajectories/{site}_relax.traj")

print("\nTo view final structures:")
print("  ase gui structures/*_final.xyz")

print("\nTo query database:")
print("  from ase.db import connect")
print("  db = connect('database/hg_on_au.db')")
print("  for row in db.select(site='fcc'):")
print("      print(row.adsorption_energy)")

print("\nTo view EOS fit:")
print("  xdg-open eos_data/eos_fit.pdf  (Linux)")
print("  open eos_data/eos_fit.pdf      (macOS)")
print("  start eos_data/eos_fit.pdf     (Windows)")

# ==============================================
# 12. Summary and Next Steps
# ==============================================

print("\n" + "="*70)
print("MIGRATION COMPLETE: What We Achieved")
print("="*70)
print("""
✓ Replaced Open Babel file reading with ASE's fcc111 builder
✓ Replaced manual atom positioning with add_adsorbate
✓ Replaced Open Babel force fields with ASE calculators (EMT/LJ)
✓ Added ASE constraints (FixAtoms) for realistic surface calculations
✓ Used ASE optimizers (BFGS/LBFGS) instead of Open Babel's internal optimizer
✓ Saved and visualized trajectories with ASE GUI
✓ Correctly computed adsorption energy with isolated atom in large box
✓ Tested multiple adsorption sites automatically
✓ Switched to ML potentials (CHGNet/MACE) with one line change
✓ Exported structures in multiple formats (XYZ, CIF, VASP)
✓ Stored results in ASE database for easy analysis
✓ Added parallel processing capability
✓ Bonus: Computed equation of state for bulk gold

Next steps:
- Try different calculators (CHGNet, MACE, DFT)
- Study molecular adsorbates (CO, H2O)
- Explore reaction pathways with NEB
- Add temperature effects with MD
""")
print("="*70)

# Optional: Final visualization of best site
if enable_visualization and best_site:
    print(f"\nFinal visualization of most stable site ({best_site})...")
    view(atoms_dict[best_site])
    if save_images:
        write(f'structures/BEST_SITE_{best_site}.png', atoms_dict[best_site], 
              rotation='90x,90y')

print("\nHappy experimenting with ASE!")