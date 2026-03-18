#!/usr/bin/env python
"""
ASE script for automated screening of adsorption sites on Au(111)
Modified version with comprehensive features for learning ASE programming
"""

import os
import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT  # MODIFICATION 3: Alternative calculator
from ase.optimize import BFGS, LBFGS  # MODIFICATION 6: Different optimizers
from ase.constraints import FixAtoms
from ase.io import write, read
from ase.calculators.mixing import SumCalculator
from ase.neighborlist import NeighborList  # MODIFICATION 13: For better site detection
from ase.db import connect  # MODIFICATION 14: For database storage
from multiprocessing import Pool  # MODIFICATION 15: For parallel processing
import argparse

# ==============================================
# Create output directories
# ==============================================
os.makedirs('vasp_outputs', exist_ok=True)
os.makedirs('ase_outputs', exist_ok=True)
os.makedirs('traj_files', exist_ok=True)
os.makedirs('xyz_outputs', exist_ok=True)  # MODIFICATION 12: New directory for XYZ files

# ==============================================
# 1. Load or Create Slab
# ==============================================

# MODIFICATION: Option to use ASE-built slab instead of hard-coded positions
use_built_slab = True  # Set to False to use the hard-coded slab

if use_built_slab:
    print("Building Au(111) slab with ASE...")
    slab = fcc111('Au', size=(3, 3, 4), vacuum=10.0, a=4.08)
else:
    # Original hard-coded slab
    slab = Atoms(
        symbols=['Au']*48,
        positions=[ # positions in Angstrom 
            [2.162792418, 0.416273963, 5.000027010],
            [-0.720892382, 2.081170159, 7.354685128],
            [0.720950007, 1.248722043, 9.709145886],
            [5.046592448, 0.416273963, 5.000027010],
            [2.162907605, 2.081170159, 7.354685128],
            [3.604750037, 1.248722043, 9.709145886],
            [7.930392134, 0.416273963, 5.000027010],
            [5.046707978, 2.081170159, 7.354685128],
            [6.488550067, 1.248722043, 9.709145886],
            [10.814192164, 0.416273963, 5.000027010],
            [7.930508008, 2.081170159, 7.354685128],
            [9.372350097, 1.248722043, 9.709145886],
            [0.720892425, 2.913718011, 5.000027010],
            [-2.162792397, 4.578614244, 7.354685128],
            [-0.720950007, 3.746166128, 9.709145886],
            [3.604692455, 2.913718011, 5.000027010],
            [0.721007590, 4.578614244, 7.354685128],
            [2.162850022, 3.746166128, 9.709145886],
            [6.488492141, 2.913718011, 5.000027010],
            [3.604807963, 4.578614244, 7.354685128],
            [5.046650052, 3.746166128, 9.709145886],
            [9.372292170, 2.913718011, 5.000027010],
            [6.488607993, 4.578614244, 7.354685128],
            [7.930450082, 3.746166128, 9.709145886],
            [-0.721007762, 5.411162394, 5.000027010],
            [-3.604692240, 7.076058032, 7.354685128],
            [-2.162850022, 6.243610213, 9.709145886],
            [2.162792268, 5.411162394, 5.000027010],
            [-0.720892253, 7.076058032, 7.354685128],
            [0.720950007, 6.243610213, 9.709145886],
            [5.046591954, 5.411162394, 5.000027010],
            [2.162908120, 7.076058032, 7.354685128],
            [3.604750037, 6.243610213, 9.709145886],
            [7.930391984, 5.411162394, 5.000027010],
            [5.046708150, 7.076058032, 7.354685128],
            [6.488550067, 6.243610213, 9.709145886],
            [-2.162907777, 7.908606479, 5.000027010],
            [-5.046592255, 9.573502117, 7.354685128],
            [-3.604750037, 8.741054298, 9.709145886],
            [0.720892253, 7.908606479, 5.000027010],
            [-2.162792268, 9.573502117, 7.354685128],
            [-0.720950007, 8.741054298, 9.709145886],
            [3.604691939, 7.908606479, 5.000027010],
            [0.721008106, 9.573502117, 7.354685128],
            [2.162850022, 8.741054298, 9.709145886],
            [6.488491969, 7.908606479, 5.000027010],
            [3.604808135, 9.573502117, 7.354685128],
            [5.046650052, 8.741054298, 9.709145886]
        ],
        cell=[ # values in Angstrom
            [11.5352001190, 0.0000000000, 0.0000000000],
            [-5.7676000595, 9.9897763408, 0.0000000000],
            [0.0000000000, 0.0000000000, 19.7091999054]
        ],
        pbc=[True, True, True]
    )

# ==============================================
# 2. Parameters (MODIFICATIONS 1, 2, 4, 5)
# ==============================================

# MODIFICATION 1: Change adsorbate element
adsorbate_symbol = 'Hg'  # Try 'O', 'S', 'C', 'N'

# MODIFICATION 2: Adjust Lennard-Jones parameters
lj_epsilon = 0.03        # Try 0.05 for stronger attraction
lj_sigma = 2.80          # Try 2.50 for smaller equilibrium distance
lj_cutoff = 10.0         # Cutoff radius

# MODIFICATION 3: Choose calculator type
use_emt = False  # Set to True to use EMT instead of LJ
use_dftd4 = False  # Set to True to use DFTD4 (if installed)

# MODIFICATION 4: Site-finding cutoffs
top_layer_cutoff = 0.5   # adjust depending on top layer thickness
bridge_cutoff = 2.9      # set slightly larger than in-plane distance - try 3.0
hollow_cutoff_min = 2.0  # try 1.8
hollow_cutoff_max = 4.0  # try 4.2
min_triangle_area = 0.1  # minimum area for hollow site detection

# MODIFICATION 5: Initial adsorption heights
top_height = 2.5         # try 3.0
bridge_height = 2.0      # try 2.2
hollow_height = 1.8      # try 2.0

# MODIFICATION 7: Force convergence criterion
fmax_convergence = 0.05  # try 0.01 for tighter, 0.1 for looser

# MODIFICATION 6: Optimizer choice
use_lbfgs = False  # Set to True to use LBFGS instead of BFGS

# ==============================================
# 3. Calculate reference energies (MODIFICATION 11)
# ==============================================

print("\n" + "="*60)
print(f"ADSORPTION SCREENING: {adsorbate_symbol} on Au(111)")
print("="*60)

# Calculate clean slab energy
print("\nCalculating clean slab energy...")
clean_slab = slab.copy()
if use_emt:
    from ase.calculators.emt import EMT
    clean_slab.calc = EMT()
else:
    clean_slab.calc = LennardJones(epsilon=lj_epsilon, sigma=lj_sigma, rc=lj_cutoff)
e_slab = clean_slab.get_potential_energy()
print(f"  Clean slab energy: {e_slab:.6f} eV")

# Calculate isolated adsorbate atom energy
print(f"\nCalculating isolated {adsorbate_symbol} atom energy...")
ads_atom = Atoms(adsorbate_symbol, pbc=True)
ads_atom.set_cell([20, 20, 20])  # Large box
ads_atom.center()
if use_emt:
    ads_atom.calc = EMT()
else:
    ads_atom.calc = LennardJones(epsilon=lj_epsilon, sigma=lj_sigma, rc=lj_cutoff)
e_atom = ads_atom.get_potential_energy()
print(f"  Isolated atom energy: {e_atom:.6f} eV")

# ==============================================
# 4. Define all unique adsorption sites (MODIFICATION 13 - improved with NeighborList)
# ==============================================

def get_adsorption_sites(slab, use_neighborlist=True):
    """
    Identify all unique adsorption sites on the slab.
    
    Args:
        slab: ASE Atoms object
        use_neighborlist: If True, use ASE NeighborList for better detection
    """
    sites = {}
    
    # Identify top layer atoms
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    top_layer_threshold = np.max(z_coords) - top_layer_cutoff
    top_layer_indices = [i for i, z in enumerate(z_coords) if z >= top_layer_threshold]
    
    print(f"\nFound {len(top_layer_indices)} top layer atoms: {[i+1 for i in top_layer_indices]}")
    
    if use_neighborlist and len(slab) < 100:
        # MODIFICATION 13: Use ASE NeighborList for better site detection
        print("Using ASE NeighborList for site detection...")
        nl = NeighborList([bridge_cutoff] * len(slab), 
                          self_interaction=False, 
                          bothways=True)
        nl.update(slab)
        
        # 1. Top sites
        for i in top_layer_indices:
            sites[f'top_{i+1}'] = {
                'position': slab[i].position[:2],
                'height': top_height,
                'atom_indices': [i],
                'type': 'top'
            }
        
        # 2. Bridge sites using NeighborList
        print("\nDetecting bridge sites:")
        for i in top_layer_indices:
            neighbors, offsets = nl.get_neighbors(i)
            for j_idx, j in enumerate(neighbors):
                if j in top_layer_indices and j > i:  # avoid duplicates
                    # Get MIC-corrected vector
                    v_ij = slab.get_distance(i, j, mic=True, vector=True)
                    pos_i = slab[i].position
                    pos_j = pos_i + v_ij
                    
                    site_name = f'bridge_{i+1}-{j+1}'
                    if site_name not in sites:
                        sites[site_name] = {
                            'position': [(pos_i[0] + pos_j[0])/2, (pos_i[1] + pos_j[1])/2],
                            'height': bridge_height,
                            'atom_indices': [i, j],
                            'type': 'bridge'
                        }
                        print(f"  Found bridge: {i+1}-{j+1}")
        
        # 3. Hollow sites (more complex - need triangles)
        print("\nDetecting hollow sites:")
        for i, idx_i in enumerate(top_layer_indices):
            for j, idx_j in enumerate(top_layer_indices):
                if idx_j <= idx_i:
                    continue
                
                dist_ij = slab.get_distance(idx_i, idx_j, mic=True)
                if dist_ij > hollow_cutoff_max:
                    continue
                
                for k, idx_k in enumerate(top_layer_indices):
                    if idx_k <= idx_j:
                        continue
                    
                    dist_jk = slab.get_distance(idx_j, idx_k, mic=True)
                    dist_ki = slab.get_distance(idx_k, idx_i, mic=True)
                    
                    # Check if all distances are reasonable for a hollow site
                    if (hollow_cutoff_min < dist_ij < hollow_cutoff_max and 
                        hollow_cutoff_min < dist_jk < hollow_cutoff_max and 
                        hollow_cutoff_min < dist_ki < hollow_cutoff_max):
                        
                        # Get MIC-corrected vectors
                        v_ij = slab.get_distance(idx_i, idx_j, mic=True, vector=True)
                        v_ik = slab.get_distance(idx_i, idx_k, mic=True, vector=True)
                        
                        # Calculate triangle area
                        cross_product = np.cross(v_ij, v_ik)
                        triangle_area = 0.5 * np.linalg.norm(cross_product)
                        
                        if triangle_area > min_triangle_area:
                            pos_i = slab[idx_i].position
                            pos_j = pos_i + v_ij
                            pos_k = pos_i + v_ik
                            
                            site_name = f'hollow_{idx_i+1}-{idx_j+1}-{idx_k+1}'
                            sites[site_name] = {
                                'position': [(pos_i[0]+pos_j[0]+pos_k[0])/3, 
                                           (pos_i[1]+pos_j[1]+pos_k[1])/3],
                                'height': hollow_height,
                                'atom_indices': [idx_i, idx_j, idx_k],
                                'type': 'hollow'
                            }
                            print(f"  Found hollow: {idx_i+1}-{idx_j+1}-{idx_k+1} (area: {triangle_area:.3f} Å²)")
    
    else:
        # Original site detection logic (kept for compatibility)
        print("Using original site detection method...")
        
        # 1. Top sites
        for i in top_layer_indices:
            sites[f'top_{i+1}'] = {
                'position': slab[i].position[:2],
                'height': top_height,
                'atom_indices': [i],
                'type': 'top'
            }
        
        # 2. Bridge sites
        for i in top_layer_indices:
            neighbors_found = []
            for j in top_layer_indices:
                if j <= i:
                    continue
                dist = slab.get_distance(i, j, mic=True)
                if dist < bridge_cutoff:
                    neighbors_found.append(j+1)
                    v_ij = slab.get_distance(i, j, mic=True, vector=True)
                    pos_i = slab[i].position
                    pos_j = pos_i + v_ij
                    sites[f'bridge_{i+1}-{j+1}'] = {
                        'position': [(pos_i[0] + pos_j[0])/2, (pos_i[1] + pos_j[1])/2],
                        'height': bridge_height,
                        'atom_indices': [i, j],
                        'type': 'bridge'
                    }
            if neighbors_found:
                print(f"  Atom {i+1}: bridges with {neighbors_found}")
        
        # 3. Hollow sites
        for i, idx_i in enumerate(top_layer_indices):
            for j, idx_j in enumerate(top_layer_indices):
                if idx_j <= idx_i:
                    continue
                dist_ij = slab.get_distance(idx_i, idx_j, mic=True)
                if dist_ij > hollow_cutoff_max:
                    continue
                for k, idx_k in enumerate(top_layer_indices):
                    if idx_k <= idx_j:
                        continue
                    dist_jk = slab.get_distance(idx_j, idx_k, mic=True)
                    dist_ki = slab.get_distance(idx_k, idx_i, mic=True)
                    
                    if (hollow_cutoff_min < dist_ij < hollow_cutoff_max and 
                        hollow_cutoff_min < dist_jk < hollow_cutoff_max and 
                        hollow_cutoff_min < dist_ki < hollow_cutoff_max):
                        
                        v_ij = slab.get_distance(idx_i, idx_j, mic=True, vector=True)
                        v_ik = slab.get_distance(idx_i, idx_k, mic=True, vector=True)
                        cross_product = np.cross(v_ij, v_ik)
                        triangle_area = 0.5 * np.linalg.norm(cross_product)
                        
                        if triangle_area > min_triangle_area:
                            pos_i = slab[idx_i].position
                            pos_j = pos_i + v_ij
                            pos_k = pos_i + v_ik
                            print(f"  Hollow: {idx_i+1}-{idx_j+1}-{idx_k+1} "
                                  f"(distances: {dist_ij:.3f}, {dist_jk:.3f}, {dist_ki:.3f} Å, "
                                  f"area: {triangle_area:.3f} Å²)")
                            
                            sites[f'hollow_{idx_i+1}-{idx_j+1}-{idx_k+1}'] = {
                                'position': [(pos_i[0]+pos_j[0]+pos_k[0])/3, 
                                           (pos_i[1]+pos_j[1]+pos_k[1])/3],
                                'height': hollow_height,
                                'atom_indices': [idx_i, idx_j, idx_k],
                                'type': 'hollow'
                            }
    
    return sites

# ==============================================
# 5. Screening with calculator
# ==============================================

def screen_site(name, site):
    """
    Screen a single adsorption site.
    
    Args:
        name: site name
        site: site dictionary with position, height, etc.
    
    Returns:
        tuple: (name, adsorption_energy, total_energy, slab_with_ads)
    """
    print(f"\n{'─'*50}")
    print(f"Site {name} ({site.get('type', 'unknown')})")
    
    # Create slab with adsorbate
    slab_with_ads = slab.copy()
    add_adsorbate(slab_with_ads, Atoms(adsorbate_symbol), 
                  height=site['height'], 
                  position=site['position'])
    
    # MODIFICATION 10: Freeze only bottom layers instead of whole slab
    freeze_entire_slab = False  # Set to False to freeze only bottom layers
    
    if freeze_entire_slab:
        # Original: freeze entire slab
        freeze_indices = list(range(len(slab)))
        print(f"  Freezing entire slab ({len(freeze_indices)} atoms)")
    else:
        # MODIFICATION 10: Freeze only bottom layers
        z_coords = slab.get_positions()[:, 2]
        bottom_threshold = np.min(z_coords) + 2.0  # adjust based on layer thickness
        freeze_indices = [i for i, z in enumerate(z_coords) if z <= bottom_threshold]
        print(f"  Freezing {len(freeze_indices)} bottom atoms (z ≤ {bottom_threshold:.2f} Å)")
    
    constraint = FixAtoms(indices=freeze_indices)
    slab_with_ads.set_constraint(constraint)
    
    # MODIFICATION 9: Print adsorbate info
    ads_index = len(slab)
    print(f"  {adsorbate_symbol} atom index: {ads_index+1} (1-based)")
    
    # MODIFICATION 3: Setup calculator
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
    
    # MODIFICATION 6 & 7: Relax with chosen optimizer and convergence criterion
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
    
    # MODIFICATION 9: Print final forces and adsorbate height
    forces = slab_with_ads.get_forces()
    ads_force = np.linalg.norm(forces[ads_index])
    print(f"  Max force on adsorbate: {ads_force:.4f} eV/Å")
    
    final_height = (slab_with_ads.get_positions()[ads_index][2] - 
                   np.max(slab.get_positions()[:,2]))
    print(f"  Final height above surface: {final_height:.3f} Å")
    
    # MODIFICATION 11: Store adsorption energy
    print(f"  Adsorption energy: {e_ads:.6f} eV")
    
    # Export structures (MODIFICATION 12)
    write(f'vasp_outputs/{adsorbate_symbol.lower()}_{name}.vasp', 
          slab_with_ads, format='vasp', direct=True)
    write(f'xyz_outputs/{adsorbate_symbol.lower()}_{name}.xyz', 
          slab_with_ads, format='xyz')
    
    return name, e_ads, e_total, slab_with_ads

# ==============================================
# 6. Main execution
# ==============================================

if __name__ == "__main__":
    # Parse command line arguments for parallel execution
    parser = argparse.ArgumentParser(description='Screen adsorption sites')
    parser.add_argument('--parallel', action='store_true', 
                       help='Run in parallel mode')
    parser.add_argument('--sites', type=int, default=None,
                       help='Number of sites to process (for testing)')
    args = parser.parse_args()
    
    # Get all adsorption sites
    sites = get_adsorption_sites(slab, use_neighborlist=True)
    site_items = list(sites.items())
    
    # Limit sites for testing if requested
    if args.sites:
        site_items = site_items[:args.sites]
    
    print(f"\nTesting {len(site_items)} adsorption sites...")
    
    # MODIFICATION 15: Parallel execution
    if args.parallel:
        print("Running in parallel mode...")
        with Pool() as pool:
            results_list = pool.map(lambda item: screen_site(item[0], item[1]), 
                                   site_items)
        # Convert results to dictionary
        energies = {name: (e_ads, e_total, atoms) 
                   for name, e_ads, e_total, atoms in results_list}
    else:
        # Serial execution
        energies = {}
        for name, site in site_items:
            name, e_ads, e_total, slab_ads = screen_site(name, site)
            energies[name] = (e_ads, e_total, slab_ads)
    
    # MODIFICATION 14: Store results in ASE database (FIXED VERSION)
    print("\nSaving results to database...")
    db = connect('adsorption_screening.db')
    
    # Helper function to clean site name for database
    def clean_site_name(name):
        """Convert site name to a valid database key"""
        return name.replace('-', '_').replace(' ', '_')
    
    for name, (e_ads, e_total, atoms) in energies.items():
        site_info = sites[name]
        clean_name = clean_site_name(name)
        
        # Prepare key-value pairs with safe keys and values
        kvp = {
            'energy_total': float(e_total),
            'energy_ads': float(e_ads),
            'site_name': str(name),
            'site_type': str(site_info.get('type', 'unknown')),
            'adsorbate': str(adsorbate_symbol),
            'calc_type': 'EMT' if use_emt else 'LJ',  # FIXED: using calc_type instead of calculator
            'epsilon': float(lj_epsilon),
            'sigma': float(lj_sigma)
        }
        
        try:
            db.write(atoms, **kvp)
            print(f"  Saved {clean_name} to database")
        except Exception as e:
            print(f"  Warning: Could not save {name} to database: {e}")
    
    print(f"\nResults saved to 'adsorption_screening.db'")
    
    # Find best site
    e_ads_values = {name: e_ads for name, (e_ads, _, _) in energies.items()}
    min_energy = min(e_ads_values.values())
    best_sites = [name for name, e in e_ads_values.items() if abs(e - min_energy) < 1e-6]
    
    # Write summary file
    with open('adsorption_summary.txt', 'w') as f:
        f.write(f"Adsorption Screening Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Adsorbate: {adsorbate_symbol}\n")
        f.write(f"Calculator: {'EMT' if use_emt else 'LJ'}\n")
        f.write(f"LJ parameters: epsilon={lj_epsilon}, sigma={lj_sigma}\n")
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
    print("\nComplete energy table (sorted by stability):")
    print("-"*60)
    print(f"{'Site':30s} {'Type':10s} {'Energy (eV)':15s}")
    print("-"*60)
    
    for name, e_ads in sorted(e_ads_values.items(), key=lambda x: x[1]):
        site_type = sites[name].get('type', 'unknown')
        stability = " ✓" if abs(e_ads - min_energy) < 1e-6 else ""
        print(f"{name:30s} {site_type:10s} {e_ads:15.6f}{stability}")
    
    print("="*60)
    print("\n" + "="*60)
    print("VISUALIZATION INSTRUCTIONS")
    print("="*60)
    print("To view trajectories:")
    print("  ase gui traj_files/hg_*.traj")
    print("\nTo view final structures:")
    print("  ase gui xyz_outputs/hg_*.xyz")
    print("\nTo query database:")
    print("  from ase.db import connect")
    print("  db = connect('adsorption_screening.db')")
    print("  for row in db.select(site_type='hollow'):")
    print("      print(row.site_name, row.energy_ads)")
    print("="*60)