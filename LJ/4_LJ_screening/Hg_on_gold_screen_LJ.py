#!/usr/bin/env python
"""
ASE script for automated screening of adsorption sites on Au(111)
Modified version with comprehensive features for learning ASE programming
ADDED: Active visualization with ase.visualize
"""

import os
import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from ase.optimize import BFGS, LBFGS
from ase.constraints import FixAtoms
from ase.io import write, read
from ase.visualize import view  # ADDED: for interactive visualization
from ase.calculators.mixing import SumCalculator
from ase.neighborlist import NeighborList
from ase.db import connect
from multiprocessing import Pool
import argparse
import time

# ==============================================
# Create output directories
# ==============================================
os.makedirs('vasp_outputs', exist_ok=True)
os.makedirs('ase_outputs', exist_ok=True)
os.makedirs('traj_files', exist_ok=True)
os.makedirs('xyz_outputs', exist_ok=True)
os.makedirs('png_outputs', exist_ok=True)  # ADDED: for saving images

# ==============================================
# 1. Load or Create Slab
# ==============================================

use_built_slab = True

if use_built_slab:
    print("Building Au(111) slab with ASE...")
    slab = fcc111('Au', size=(3, 3, 4), vacuum=10.0, a=4.08)
else:
    # Original hard-coded slab (truncated for brevity)
    slab = Atoms(...)  # full slab definition here

# ==============================================
# 2. Parameters
# ==============================================

adsorbate_symbol = 'Hg'
lj_epsilon = 0.03
lj_sigma = 2.80
lj_cutoff = 10.0
use_emt = False
use_dftd4 = False

top_layer_cutoff = 0.5
bridge_cutoff = 2.9
hollow_cutoff_min = 2.0
hollow_cutoff_max = 4.0
min_triangle_area = 0.1

top_height = 2.5
bridge_height = 2.0
hollow_height = 1.8

fmax_convergence = 0.05
use_lbfgs = False

# VISUALIZATION SETTINGS
enable_visualization = True  # ADDED: toggle visualization on/off
save_images = True           # ADDED: save PNG images of structures
visualize_every_step = False # ADDED: show each site during screening

# ==============================================
# 3. Calculate reference energies
# ==============================================

print("\n" + "="*60)
print(f"ADSORPTION SCREENING: {adsorbate_symbol} on Au(111)")
print("="*60)

print("\nCalculating clean slab energy...")
clean_slab = slab.copy()
if use_emt:
    from ase.calculators.emt import EMT
    clean_slab.calc = EMT()
else:
    clean_slab.calc = LennardJones(epsilon=lj_epsilon, sigma=lj_sigma, rc=lj_cutoff)

# VISUALIZATION: Show clean slab
if enable_visualization:
    print("  Visualizing clean slab...")
    view(clean_slab)
    if save_images:
        write('png_outputs/clean_slab.png', clean_slab, rotation='90x,90y')

e_slab = clean_slab.get_potential_energy()
print(f"  Clean slab energy: {e_slab:.6f} eV")

print(f"\nCalculating isolated {adsorbate_symbol} atom energy...")
ads_atom = Atoms(adsorbate_symbol, pbc=True)
ads_atom.set_cell([20, 20, 20])
ads_atom.center()
if use_emt:
    ads_atom.calc = EMT()
else:
    ads_atom.calc = LennardJones(epsilon=lj_epsilon, sigma=lj_sigma, rc=lj_cutoff)

# VISUALIZATION: Show isolated atom
if enable_visualization:
    print("  Visualizing isolated atom...")
    view(ads_atom)
    if save_images:
        write('png_outputs/isolated_atom.png', ads_atom)

e_atom = ads_atom.get_potential_energy()
print(f"  Isolated atom energy: {e_atom:.6f} eV")

# ==============================================
# 4. Define all unique adsorption sites
# ==============================================

def get_adsorption_sites(slab, use_neighborlist=True):
    """Identify all unique adsorption sites on the slab."""
    sites = {}
    
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    top_layer_threshold = np.max(z_coords) - top_layer_cutoff
    top_layer_indices = [i for i, z in enumerate(z_coords) if z >= top_layer_threshold]
    
    print(f"\nFound {len(top_layer_indices)} top layer atoms: {[i+1 for i in top_layer_indices]}")
    
    # ... (site detection code - unchanged) ...
    
    return sites

# ==============================================
# 5. Screening with calculator and visualization
# ==============================================

def visualize_structure(atoms, title, site_name=""):
    """Helper function for consistent visualization"""
    if not enable_visualization:
        return
    
    print(f"  Visualizing: {title}")
    view(atoms)
    
    if save_images:
        timestamp = time.strftime("%H%M%S")
        filename = f'png_outputs/{site_name}_{title}_{timestamp}.png'
        write(filename, atoms, rotation='90x,90y')

def screen_site(name, site):
    """
    Screen a single adsorption site with visualization.
    """
    print(f"\n{'─'*50}")
    print(f"Site {name} ({site.get('type', 'unknown')})")
    
    # Create slab with adsorbate
    slab_with_ads = slab.copy()
    add_adsorbate(slab_with_ads, Atoms(adsorbate_symbol), 
                  height=site['height'], 
                  position=site['position'])
    
    # VISUALIZATION: Initial configuration
    if visualize_every_step:
        visualize_structure(slab_with_ads, "initial", name)
    
    # Freeze bottom layers
    z_coords = slab.get_positions()[:, 2]
    bottom_threshold = np.min(z_coords) + 2.0
    freeze_indices = [i for i, z in enumerate(z_coords) if z <= bottom_threshold]
    print(f"  Freezing {len(freeze_indices)} bottom atoms (z ≤ {bottom_threshold:.2f} Å)")
    
    constraint = FixAtoms(indices=freeze_indices)
    slab_with_ads.set_constraint(constraint)
    
    ads_index = len(slab)
    print(f"  {adsorbate_symbol} atom index: {ads_index+1} (1-based)")
    
    # Setup calculator
    if use_emt:
        from ase.calculators.emt import EMT
        slab_with_ads.calc = EMT()
    elif use_dftd4:
        try:
            from dftd4.ase import DFTD4
            slab_with_ads.calc = SumCalculator([
                LennardJones(epsilon=lj_epsilon, sigma=lj_sigma, rc=lj_cutoff),
                DFTD4(method="PBE")
            ])
        except ImportError:
            print("  Warning: DFTD4 not installed, using LJ only")
            slab_with_ads.calc = LennardJones(epsilon=lj_epsilon, sigma=lj_sigma, rc=lj_cutoff)
    else:
        slab_with_ads.calc = LennardJones(epsilon=lj_epsilon, sigma=lj_sigma, rc=lj_cutoff)
    
    # Relax with trajectory
    traj_file = f'traj_files/{adsorbate_symbol.lower()}_{name}.traj'
    
    if use_lbfgs:
        from ase.optimize import LBFGS
        relax = LBFGS(slab_with_ads, trajectory=traj_file)
    else:
        relax = BFGS(slab_with_ads, trajectory=traj_file)
    
    relax.run(fmax=fmax_convergence)
    
    # Get energies
    e_total = slab_with_ads.get_potential_energy()
    e_ads = e_total - (e_slab + e_atom)
    
    # Print forces and height
    forces = slab_with_ads.get_forces()
    ads_force = np.linalg.norm(forces[ads_index])
    print(f"  Max force on adsorbate: {ads_force:.4f} eV/Å")
    
    final_height = (slab_with_ads.get_positions()[ads_index][2] - 
                   np.max(slab.get_positions()[:,2]))
    print(f"  Final height above surface: {final_height:.3f} Å")
    print(f"  Adsorption energy: {e_ads:.6f} eV")
    
    # VISUALIZATION: Final configuration
    if visualize_every_step:
        visualize_structure(slab_with_ads, "final", name)
    
    # Export structures
    write(f'vasp_outputs/{adsorbate_symbol.lower()}_{name}.vasp', 
          slab_with_ads, format='vasp', direct=True)
    write(f'xyz_outputs/{adsorbate_symbol.lower()}_{name}.xyz', 
          slab_with_ads, format='xyz')
    
    # VISUALIZATION: Save final image
    if save_images:
        write(f'png_outputs/{name}_final.png', slab_with_ads, rotation='90x,90y')
    
    return name, e_ads, e_total, slab_with_ads

# ==============================================
# 6. Main execution
# ==============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Screen adsorption sites')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel mode')
    parser.add_argument('--sites', type=int, default=None, help='Number of sites to process')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--viz-all', action='store_true', help='Visualize every site')
    args = parser.parse_args()
    
    # Update visualization settings from command line
    if args.no_viz:
        enable_visualization = False
    if args.viz_all:
        visualize_every_step = True
    
    # Get all adsorption sites
    sites = get_adsorption_sites(slab, use_neighborlist=True)
    site_items = list(sites.items())
    
    if args.sites:
        site_items = site_items[:args.sites]
    
    print(f"\nTesting {len(site_items)} adsorption sites...")
    print(f"Visualization: {'ON' if enable_visualization else 'OFF'}")
    if enable_visualization:
        print("  - Windows will pop up showing structures")
        print("  - Close each window to continue")
        if visualize_every_step:
            print("  - Visualizing EVERY site (may take time)")
    
    # Run screening
    if args.parallel:
        print("Running in parallel mode...")
        with Pool() as pool:
            results_list = pool.map(lambda item: screen_site(item[0], item[1]), site_items)
        energies = {name: (e_ads, e_total, atoms) for name, e_ads, e_total, atoms in results_list}
    else:
        energies = {}
        for i, (name, site) in enumerate(site_items):
            print(f"\nProgress: Site {i+1}/{len(site_items)}")
            name, e_ads, e_total, slab_ads = screen_site(name, site)
            energies[name] = (e_ads, e_total, slab_ads)
    
    # Save to database
    print("\nSaving results to database...")
    db = connect('adsorption_screening.db')
    
    def clean_site_name(name):
        return name.replace('-', '_').replace(' ', '_')
    
    for name, (e_ads, e_total, atoms) in energies.items():
        site_info = sites[name]
        clean_name = clean_site_name(name)
        
        kvp = {
            'energy_total': float(e_total),
            'energy_ads': float(e_ads),
            'site_name': str(name),
            'site_type': str(site_info.get('type', 'unknown')),
            'adsorbate': str(adsorbate_symbol),
            'calc_type': 'EMT' if use_emt else 'LJ',
            'epsilon': float(lj_epsilon),
            'sigma': float(lj_sigma)
        }
        
        try:
            db.write(atoms, **kvp)
            print(f"  Saved {clean_name} to database")
        except Exception as e:
            print(f"  Warning: Could not save {name} to database: {e}")
    
    # Find best site
    e_ads_values = {name: e_ads for name, (e_ads, _, _) in energies.items()}
    min_energy = min(e_ads_values.values())
    best_sites = [name for name, e in e_ads_values.items() if abs(e - min_energy) < 1e-6]
    
    # Write summary
    with open('adsorption_summary.txt', 'w') as f:
        f.write(f"Adsorption Screening Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Adsorbate: {adsorbate_symbol}\n")
        f.write(f"Calculator: {'EMT' if use_emt else 'LJ'}\n")
        f.write(f"Number of sites tested: {len(sites)}\n")
        f.write(f"Most stable configuration(s): {', '.join(best_sites)}\n")
        f.write(f"Minimum energy: {min_energy:.6f} eV\n\n")
        f.write(f"{'Site':30s} {'Type':10s} {'Energy (eV)':15s}\n")
        f.write("-"*60 + "\n")
        
        for name, e_ads in sorted(e_ads_values.items(), key=lambda x: x[1]):
            site_type = sites[name].get('type', 'unknown')
            f.write(f"{name:30s} {site_type:10s} {e_ads:15.6f}\n")
            if abs(e_ads - min_energy) < 1e-6:
                f.write(" " * 30 + " <--- Most stable\n")
    
    # Console output
    print("\n" + "="*60)
    print("SCREENING COMPLETE")
    print("="*60)
    print(f"Most stable configuration(s): {', '.join(best_sites)}")
    print(f"Minimum energy: {min_energy:.6f} eV")
    
    # FINAL VISUALIZATION: Best site
    if enable_visualization and best_sites:
        print("\nVisualizing most stable configuration...")
        best_name = best_sites[0]
        best_atoms = energies[best_name][2]
        view(best_atoms)
        if save_images:
            write(f'png_outputs/BEST_SITE_{best_name}.png', best_atoms, rotation='90x,90y')
    
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print("Interactive visualization windows were shown during the run.")
    print("\nTo view saved trajectories:")
    print("  ase gui traj_files/*.traj")
    print("\nTo view saved images:")
    print("  ls png_outputs/")
    print("\nTo view final structures:")
    print("  ase gui xyz_outputs/*.xyz")
    print("\nTo load and visualize in Python:")
    print("  from ase.io import read")
    print("  from ase.visualize import view")
    print("  atoms = read('xyz_outputs/hg_top_1.xyz')")
    print("  view(atoms)")
    print("="*60)