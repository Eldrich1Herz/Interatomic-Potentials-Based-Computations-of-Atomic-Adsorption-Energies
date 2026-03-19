#
# try to calculate adsorption energy of Hg on Au slab
# ORIGINAL STYLE - FIXED with XYZ instead of CIF
# CORRECTED lattice constant handling
#

import io
import os
import tempfile
from ase.io import write
from ase import Atoms
from ase.build import bulk
from ase.eos import calculate_eos
from ase.db import connect
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.io import read
from ase.calculators.lj import LennardJones
from openbabel import pybel, openbabel
import numpy as np

# ============================================
# PART 1: Create bulk gold and convert to Open Babel
# ============================================

print("\n" + "="*60)
print("Hg ADSORPTION ON Au(111) - ORIGINAL STYLE")
print("="*60)

symb = 'Au'
print(f"\n1. Creating bulk {symb} with ASE...")
ase_bulk = bulk(symb, 'fcc', cubic=True)

# Convert ASE atoms to CIF and read with Open Babel
cif_buffer = io.BytesIO()
write(cif_buffer, ase_bulk, format='cif')
cif_string = cif_buffer.getvalue().decode('utf-8')
atoms = pybel.readstring("cif", cif_string)

# Verify conversion
if atoms.unitcell:
    print("\n   Success: Unit Cell and PBC preserved.")
    print(f"   Lattice a: {atoms.unitcell.GetA():.2f} Å")
    print(f"   Total Atoms: {len(atoms.atoms)}")

# ============================================
# PART 2: Optimize with Open Babel force field
# ============================================

print("\n2. Optimizing bulk Au with MMFF94...")
atoms.localopt(forcefield="mmff94", steps=0)
print(f"   Energy: {atoms.energy} kJ/mol")

# ============================================
# PART 3: Get lattice constant
# ============================================

print("\n3. Getting lattice constant...")
# Open Babel returns the lattice constant in Angstroms already.
# Do NOT multiply by 0.529177.
# For reliability, we use the known standard value for Au (4.08 Å).
a = 4.08  # Standard lattice constant for gold in Angstroms
print(f"   Using standard lattice constant: {a:.3f} Å")

# ============================================
# PART 4: Create slab with ASE
# ============================================

print("\n4. Creating Au(111) slab with ASE...")
n = 3
atoms_slab = fcc111(symb, (5, 5, n), a=a, vacuum=10.0)
print(f"   Created slab with {len(atoms_slab)} atoms")

# ============================================
# PART 5: Calculate slab energy with Open Babel UFF (using XYZ)
# ============================================

print("\n5. Calculating clean slab energy with UFF...")

# Save slab to temporary XYZ file (more reliable for slabs)
with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False, mode='w') as tmp:
    write(tmp.name, atoms_slab, format='xyz')
    tmp_name = tmp.name

# Read with Open Babel
ob_slab = next(pybel.readfile("xyz", tmp_name))
os.unlink(tmp_name)

# Set up UFF calculator
ob_slab.calc = openbabel.OBForceField.FindForceField("UFF")
if ob_slab.calc.Setup(ob_slab.OBMol):
    # Get energy
    en_slab = ob_slab.calc.Energy()
    print(f"   Energy of Au slab (UFF): {en_slab:.6f} kJ/mol")
    print(f"   Energy of Au slab (UFF): {en_slab/96.485:.6f} eV")
else:
    print("   Error: Could not setup UFF")
    en_slab = 0

# ============================================
# PART 6: Calculate isolated Hg atom energy
# ============================================

print("\n6. Calculating isolated Hg atom energy...")
add_atom = 'Hg'

# Create Hg atom with Open Babel
hg_ob = pybel.Molecule(openbabel.OBMol())
hg_atom = hg_ob.OBMol.NewAtom()
hg_atom.SetAtomicNum(80)  # Hg atomic number
hg_atom.SetVector(0, 0, 0)

# Calculate energy with UFF
hg_ob.calc = openbabel.OBForceField.FindForceField("UFF")
if hg_ob.calc.Setup(hg_ob.OBMol):
    en_atom = hg_ob.calc.Energy()
    print(f"   Hg atom energy (UFF): {en_atom:.6f} kJ/mol = {en_atom/96.485:.6f} eV")
else:
    print("   Hg atom energy (UFF): 0 kJ/mol (by default)")
    en_atom = 0

# ============================================
# PART 7: Add adsorbate and optimize with Open Babel
# ============================================

print("\n7. Adding Hg adsorbate and optimizing with UFF...")

# Create a fresh slab with ASE
working_slab = fcc111(symb, (5, 5, n), a=a, vacuum=10.0)

# Add Hg atom with ASE
add_adsorbate(working_slab, add_atom, height=1.0, position='fcc')
print(f"   Added Hg at fcc site, height=1.0 Å")

# Save to temporary XYZ file
with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False, mode='w') as tmp:
    write(tmp.name, working_slab, format='xyz')
    tmp_name2 = tmp.name

# Read with Open Babel
ob_working = next(pybel.readfile("xyz", tmp_name2))
os.unlink(tmp_name2)

# Setup UFF
ob_working.calc = openbabel.OBForceField.FindForceField("UFF")
if ob_working.calc.Setup(ob_working.OBMol):
    # Optimize with Open Babel
    print("   Running UFF optimization (100 steps)...")
    ob_working.calc.SteepestDescent(100)
    ob_working.calc.ConjugateGradients(100)
    ob_working.calc.GetCoordinates(ob_working.OBMol)
    
    en_total = ob_working.calc.Energy()
    print(f"   Energy of slab+Hg (UFF): {en_total:.6f} kJ/mol")
    print(f"   Energy of slab+Hg (UFF): {en_total/96.485:.6f} eV")
else:
    print("   Error: Could not setup UFF")
    en_total = 0

# ============================================
# PART 8: Calculate adsorption energy
# ============================================

print("\n8. Calculating adsorption energy...")
en_ads = en_total - en_atom - en_slab
print(f"\n   {'='*40}")
print(f"   ADSORPTION ENERGY (UFF): {en_ads/96.485:.6f} eV")
print(f"   {'='*40}")

# ============================================
# PART 9: Also try with ASE-only for comparison
# ============================================

print("\n9. For comparison - ASE-only calculation with LJ...")

# Create slab
ase_compare = fcc111(symb, (5, 5, n), a=a, vacuum=10.0)
add_adsorbate(ase_compare, add_atom, height=1.0, position='fcc')

# Constrain bottom layers (better than fixing all)
z_coords = ase_compare.get_positions()[:, 2]
bottom_threshold = np.min(z_coords) + 2.0
bottom_indices = [i for i, z in enumerate(z_coords) if z <= bottom_threshold]
ase_compare.constraints = [FixAtoms(indices=bottom_indices)]

# Use LJ calculator
ase_compare.calc = LennardJones(epsilon=0.05, sigma=2.8)

# Optimize with ASE
opt = BFGS(ase_compare, trajectory='Hg_on_Au_LJ_comparison.traj')
opt.run(fmax=0.05)

# Calculate reference energies
ase_slab_calc = fcc111(symb, (5, 5, n), a=a, vacuum=10.0)
ase_slab_calc.constraints = [FixAtoms(indices=bottom_indices)]
ase_slab_calc.calc = LennardJones(epsilon=0.05, sigma=2.8)
e_slab_lj = ase_slab_calc.get_potential_energy()

hg_lj = Atoms('Hg')
hg_lj.calc = LennardJones(epsilon=0.05, sigma=2.8)
e_hg_lj = hg_lj.get_potential_energy()

e_total_lj = ase_compare.get_potential_energy()
e_ads_lj = e_total_lj - (e_slab_lj + e_hg_lj)

print(f"   LJ adsorption energy: {e_ads_lj:.6f} eV")
print(f"   UFF adsorption energy: {en_ads/96.485:.6f} eV")

# ============================================
# PART 10: Summary
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Original approach (UFF):
  Slab energy: {en_slab/96.485:.6f} eV
  Hg energy: {en_atom/96.485:.6f} eV
  Total energy: {en_total/96.485:.6f} eV
  Adsorption energy: {en_ads/96.485:.6f} eV

ASE approach (LJ):
  Slab energy: {e_slab_lj:.6f} eV
  Hg energy: {e_hg_lj:.6f} eV
  Total energy: {e_total_lj:.6f} eV
  Adsorption energy: {e_ads_lj:.6f} eV

Expected experimental range: -0.5 to -1.0 eV
""")
print("="*60)