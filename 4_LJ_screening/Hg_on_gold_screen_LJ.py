#!/usr/bin/env python
"""
ASE script for automated screening of adsorption sites on Au(111)
Modified version with comprehensive features for learning ASE programming
FULLY FIXED VERSION: Complete site detection and visualization
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
from ase.visualize import view
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
os.makedirs('png_outputs', exist_ok=True)

# ==============================================
# 1. Load or Create Slab
# ==============================================

use_built_slab = True

if use_built_slab:
    print("Building Au(111) slab with ASE...")
    slab = fcc111('Au', size=(3, 3, 4), vacuum=10.0, a=4.08)
    print(f"Created slab with {len(slab)} atoms")

# ==============================================
# 2. Parameters
# ==============================================

adsorbate_symbol = 'Hg'
lj_epsilon = 0.03
lj_sigma = 2.80
lj_cutoff = 10.0
use_emt = False
use_dftd4 = False

# Site-finding parameters
top_layer_cutoff = 0.5
bridge_cutoff = 3.0  # Slightly larger for better detection
hollow_cutoff_min = 2.0
hollow_cutoff_max = 4.0
min_triangle_area = 0.1

# Initial heights
top_height = 2.5
bridge_height = 2.0
hollow_height = 1.8

fmax_convergence = 0.05
use_lbfgs = False

# VISUALIZATION SETTINGS
enable_visualization = True
save_images = True
visualize_every_step = False

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

if enable_visualization:
    print("  Visualizing isolated atom...")
    view(ads_atom)
    if save_images:
        write('png_outputs/isolated_atom.png', ads_atom)

e_atom = ads_atom.get_potential_energy()
print(f"  Isolated atom energy: {e_atom:.6f} eV")

# ==============================================
# 4. Define all unique adsorption sites (FULL VERSION)
# ==============================================

def get_adsorption_sites(slab, use_neighborlist=True):
    """Identify all unique adsorption sites on the slab."""
    sites = {}
    
    # Get positions
    positions = slab.get_positions()
    
    # Identify top layer atoms (highest z)
    z_coords = positions[:, 2]
    top_z = np.max(z_coords)
    top_layer_indices = [i for i, z in enumerate(z_coords) if abs(z - top_z) < 0.1]
    
    print(f"\nFound {len(top_layer_indices)} top layer atoms: {[i+1 for i in top_layer_indices]}")
    
    # 1. Top sites
    for i in top_layer_indices:
        site_name = f'top_{i+1}'
        sites[site_name] = {
            'position': positions[i, :2].copy(),
            'height': top_height,
            'type': 'top',
            'indices': [i]
        }
    
    # 2. Bridge sites
    print("\nDetecting bridge sites:")
    bridge_count = 0
    for i in top_layer_indices:
        for j in top_layer_indices:
            if j <= i:
                continue
            
            # Calculate 2D distance
            dist = np.linalg.norm(positions[i, :2] - positions[j, :2])
            
            if dist < bridge_cutoff:
                # Midpoint
                mid_point = (positions[i, :2] + positions[j, :2]) / 2
                site_name = f'bridge_{i+1}-{j+1}'
                sites[site_name] = {
                    'position': mid_point.copy(),
                    'height': bridge_height,
                    'type': 'bridge',
                    'indices': [i, j]
                }
                bridge_count += 1
                print(f"  Found bridge: {i+1}-{j+1} (dist={dist:.3f} Å)")
    
    print(f"  Total bridge sites: {bridge_count}")
    
    # 3. Hollow sites (triangles)
    print("\nDetecting hollow sites:")
    hollow_count = 0
    for i, idx_i in enumerate(top_layer_indices):
        for j, idx_j in enumerate(top_layer_indices):
            if idx_j <= idx_i:
                continue
            
            for k, idx_k in enumerate(top_layer_indices):
                if idx_k <= idx_j:
                    continue
                
                # Get positions
                p_i = positions[idx_i, :2]
                p_j = positions[idx_j, :2]
                p_k = positions[idx_k, :2]
                
                # Calculate distances
                d_ij = np.linalg.norm(p_i - p_j)
                d_jk = np.linalg.norm(p_j - p_k)
                d_ki = np.linalg.norm(p_k - p_i)
                
                # Check if triangle is reasonable
                if (hollow_cutoff_min < d_ij < hollow_cutoff_max and
                    hollow_cutoff_min < d_jk < hollow_cutoff_max and
                    hollow_cutoff_min < d_ki < hollow_cutoff_max):
                    
                    # Calculate triangle center
                    center = (p_i + p_j + p_k) / 3
                    
                    # Calculate area to check collinearity
                    area = 0.5 * abs(np.cross(p_j - p_i, p_k - p_i))
                    
                    if area > min_triangle_area:
                        site_name = f'hollow_{idx_i+1}-{idx_j+1}-{idx_k+1}'
                        sites[site_name] = {
                            'position': center.copy(),
                            'height': hollow_height,
                            'type': 'hollow',
                            'indices': [idx_i, idx_j, idx_k]
                        }
                        hollow_count += 1
                        print(f"  Found hollow: {idx_i+1}-{idx_j+1}-{idx_k+1} (area={area:.3f} Å²)")
    
    print(f"  Total hollow sites: {hollow_count}")
    print(f"\nTotal sites found: {len(sites)}")
    
    return sites

# ==============================================
# 5. Helper function for visualization
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

# ==============================================
# 6. Screen a single site
# ==============================================

def screen_site(name, site):
    """Screen a single adsorption site with visualization."""
    print(f"\n{'─'*50}")
    print(f"Site {name} ({site.get('type', 'unknown')})")
    
    # Create slab with adsorbate
    slab_with_ads = slab.copy()
    add_adsorbate(slab_with_ads, Atoms(adsorbate_symbol), 
                  height=site['height'], 
                  position=site['position'])
    
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
    
    if visualize_every_step:
        visualize_structure(slab_with_ads, "final", name)
    
    # Export structures
    write(f'vasp_outputs/{adsorbate_symbol.lower()}_{name}.vasp', 
          slab_with_ads, format='vasp', direct=True)
    write(f'xyz_outputs/{adsorbate_symbol.lower()}_{name}.xyz', 
          slab_with_ads, format='xyz')
    
    if save_images:
        write(f'png_outputs/{name}_final.png', slab_with_ads, rotation='90x,90y')
    
    return name, e_ads, e_total, slab_with_ads

# ==============================================
# 7. Main execution
# ==============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Screen adsorption sites')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel mode')
    parser.add_argument('--sites', type=int, default=None, help='Number of sites to process')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--viz-all', action='store_true', help='Visualize every site')
    args = parser.parse_args()
    
    # Update visualization settings
    if args.no_viz:
        enable_visualization = False
    if args.viz_all:
        visualize_every_step = True
    
    # Get all adsorption sites
    sites = get_adsorption_sites(slab, use_neighborlist=True)
    site_items = list(sites.items())
    
    if not site_items:
        print("\n❌ ERROR: No adsorption sites found!")
        print("Check your site detection parameters.")
        exit(1)
    
    if args.sites:
        site_items = site_items[:args.sites]
    
    print(f"\nTesting {len(site_items)} adsorption sites...")
    print(f"Visualization: {'ON' if enable_visualization else 'OFF'}")
    
    # Run screening
    energies = {}
    final_structures = {}
    
    for i, (name, site) in enumerate(site_items):
        print(f"\nProgress: Site {i+1}/{len(site_items)}")
        name, e_ads, e_total, slab_ads = screen_site(name, site)
        energies[name] = e_ads
        final_structures[name] = slab_ads
    
    # Save to database
    print("\nSaving results to database...")
    db = connect('adsorption_screening.db')
    
    for name, e_ads in energies.items():
        site_info = sites[name]
        kvp = {
            'energy_ads': float(e_ads),
            'site_name': str(name),
            'site_type': str(site_info.get('type', 'unknown')),
            'adsorbate': str(adsorbate_symbol),
            'calc_type': 'EMT' if use_emt else 'LJ',
            'epsilon': float(lj_epsilon),
            'sigma': float(lj_sigma)
        }
        
        try:
            db.write(final_structures[name], **kvp)
            print(f"  Saved {name} to database")
        except Exception as e:
            print(f"  Warning: Could not save {name}: {e}")
    
    # Find best site
    if energies:
        min_energy = min(energies.values())
        best_sites = [name for name, e in energies.items() if abs(e - min_energy) < 1e-6]
        
        print("\n" + "="*60)
        print("SCREENING COMPLETE")
        print("="*60)
        print(f"Most stable configuration(s): {', '.join(best_sites)}")
        print(f"Minimum energy: {min_energy:.6f} eV")
        
        # Final visualization
        if enable_visualization and best_sites:
            print("\nVisualizing most stable configuration...")
            view(final_structures[best_sites[0]])
            if save_images:
                write(f'png_outputs/BEST_SITE_{best_sites[0]}.png', 
                      final_structures[best_sites[0]], rotation='90x,90y')
    
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print("\nTo view trajectories:")
    print("  ase gui traj_files/*.traj")
    print("\nTo view saved images:")
    print("  dir png_outputs\\")
    print("\nTo view final structures:")
    print("  ase gui xyz_outputs/*.xyz")
    print("="*60)