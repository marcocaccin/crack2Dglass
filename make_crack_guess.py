#! /usr/bin/env python

import numpy as np
import ase.io
import ase.units as units
from matscipy.fracture_mechanics.crack import thin_strip_displacement_y
from quippy import Atoms as qpAtoms

import sys
sys.path.insert(0, '.')
import params

# now, we build system aligned with requested crystallographic orientation
crack_slab = params.unit_slab

pos_0 = np.array([-21.5, -2.9, 31.])
pos_1 = np.array([-21.4, -1.38, 30.6])
r = crack_slab.get_positions()
atom_0, atom_1 = [np.argsort([np.linalg.norm(r_i - pos) for r_i in r])[0]
                  for pos in [pos_0, pos_1]]

# center vertically half way along the vertical bond between atoms 0 and 1
crack_slab.positions[:, 1] += (crack_slab.positions[atom_1, 1] -
                              crack_slab.positions[atom_0, 1]) / 2.0

# map positions back into unit cell
crack_slab.set_scaled_positions(crack_slab.get_scaled_positions())

# open up the cell along x and y by introducing some vacuum
def shrink_cell(atoms):
    r = atoms.get_positions()
    mins = [pos.min() for pos in r.T]
    maxs = [pos.max() for pos in r.T]
    atoms.set_cell([b - a for a, b in zip(mins, maxs)])
    return

shrink_cell(crack_slab)
crack_slab.center(params.vacuum, axis=0)
crack_slab.center(params.vacuum, axis=1)

# centre the slab on the origin
crack_slab.positions[:, 0] -= crack_slab.positions[:, 0].mean()
crack_slab.positions[:, 1] -= crack_slab.positions[:, 1].mean()
crack_slab.positions[:, 1] += 0.5

# ****** Apply initial strain ramp *****

strain = params.initial_strain

left = crack_slab.positions[:, 0].min()
right = crack_slab.positions[:, 0].max()

displacement = thin_strip_displacement_y(
    crack_slab.positions[:, 0],
    crack_slab.positions[:, 1],
    strain,
    left + params.crack_seed_length,
    left + params.crack_seed_length +
    params.strain_ramp_length)

crack_slab.positions[:, 1] += displacement
# ******** Save output file **********
qpcrack = qpAtoms(crack_slab)
qpcrack.map_into_cell()
pretty = ase.atoms.Atoms(crack_slab.get_atomic_numbers(), np.array(qpcrack.positions), cell=crack_slab.get_cell())
pretty.set_array('index', np.arange(len(crack_slab)))
# Save results in extended XYZ format, including extra properties and info
ase.io.write('crack_guess.xyz', pretty, format='extxyz')
