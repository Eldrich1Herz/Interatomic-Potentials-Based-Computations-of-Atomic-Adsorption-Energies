#
# ase adsorption energy of Hg on gold slab write code
#
# Modified version with explicit LJ parameters and visualization
#

from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.visualize import view  # ADDED: for visualization
from ase.io import write, read  # ADDED: for saving/loading structures

# 1. Setup Parameters
adsorbate_element = 'Hg'
slab_element = 'Au'
vacuum_size = 10.0

# 2. Create the Au(111) Slab (4 layers, 3x3 supercell)
slab = fcc111(slab_element, size=(3, 3, 4), vacuum=vacuum_size)

# Using Lennard-Jones with explicit parameters
slab.calc = LennardJones(epsilon=0.05, sigma=2.8, rc=10.0)

# VISUALIZATION 1: View initial clean slab
print("\nVisualizing initial clean slab...")
view(slab)  # This will open a window showing the clean slab

# 3. Calculate Energy of the Clean Slab
print("Optimizing clean slab...")
dyn_slab = BFGS(slab, logfile=None, trajectory='clean_slab.traj')  # ADDED: trajectory
dyn_slab.run(fmax=0.05)
e_slab = slab.get_potential_energy()

# VISUALIZATION 2: View relaxed clean slab
print("Visualizing relaxed clean slab...")
view(slab)

# 4. Calculate Energy of the Isolated Hg Atom
atom = Atoms(adsorbate_element, pbc=True)
atom.set_cell([20, 20, 20])
atom.center()
atom.calc = LennardJones(epsilon=0.05, sigma=2.8, rc=10.0)
e_atom = atom.get_potential_energy()

print(f"\nReference energies:")
print(f"  Clean slab: {e_slab:.4f} eV")
print(f"  Isolated Hg atom: {e_atom:.4f} eV")

# 5. Test different adsorption sites
print("\n" + "="*50)
print("Testing Hg adsorption on different sites")
print("="*50)

sites = ['ontop', 'fcc', 'hcp', 'bridge']
results = {}
trajectory_files = []  # ADDED: store trajectory names

for site in sites:
    print(f"\n{'─'*40}")
    print(f"Testing {site} site...")
    
    # Create a fresh copy of the slab for each site
    slab_ads = slab.copy()
    
    # Place Hg atom
    add_adsorbate(slab_ads, adsorbate_element, height=2.2, position=site)
    
    # VISUALIZATION 3: View initial configuration for each site
    print(f"  Visualizing initial {site} configuration...")
    view(slab_ads)
    
    # Attach calculator
    slab_ads.calc = LennardJones(epsilon=0.05, sigma=2.8, rc=10.0)
    
    # Optimize with trajectory saving
    traj_file = f'hg_on_au_{site}.traj'
    trajectory_files.append(traj_file)
    
    dyn = BFGS(slab_ads, trajectory=traj_file, logfile=None)
    dyn.run(fmax=0.05)
    
    e_total = slab_ads.get_potential_energy()
    e_ads = e_total - (e_slab + e_atom)
    results[site] = e_ads
    
    print(f"  E_total = {e_total:.4f} eV, E_ads = {e_ads:8.4f} eV")
    
    # VISUALIZATION 4: View final relaxed structure
    print(f"  Visualizing final {site} configuration...")
    view(slab_ads)
    
    # EXPORT: Save final structure in multiple formats
    write(f'hg_on_au_{site}_final.xyz', slab_ads)
    write(f'hg_on_au_{site}_final.cif', slab_ads)
    print(f"  Saved final structure to hg_on_au_{site}_final.xyz and .cif")

# 6. Find the most stable site
best_site = min(results, key=results.get)
print("\n" + "-"*50)
print(f"Most stable site: {best_site} with E_ads = {results[best_site]:.4f} eV")
print("-"*50)

# 7. Visualization Instructions
print("\n" + "="*50)
print("VISUALIZATION SUMMARY")
print("="*50)
print("\nTo view trajectories with ASE GUI:")
for traj in trajectory_files:
    print(f"  ase gui {traj}")

print("\nTo view trajectories in Python:")
print("  from ase.io import read")
print("  from ase.visualize import view")
print("  traj = read('hg_on_au_fcc.traj', index=':')")
print("  view(traj)")

print("\nTo view final structures:")
print("  ase gui hg_on_au_*_final.xyz")
print("  ase gui hg_on_au_*_final.cif")

# 8. Optional: Load and view all trajectories together
print("\n" + "="*50)
print("LOADING ALL TRAJECTORIES FOR COMPARISON")
print("="*50)
try:
    from ase.io import read
    all_trajectories = []
    for traj in trajectory_files:
        traj_atoms = read(traj, index=':')
        all_trajectories.extend(traj_atoms)
    print(f"Loaded {len(all_trajectories)} structures from all trajectories")
    print("To view all structures together, use:")
    print("  view(all_trajectories)")
except Exception as e:
    print(f"Could not load trajectories: {e}")

# 9. Note about the results
print("\nNOTE: Lennard-Jones is a simple pair potential.")
print("For quantitative results, use DFT or ML potentials like CHGNet or MACE.")
print("Experimental adsorption energy for Hg on Au(111): -0.5 to -1.0 eV")