7#! /usr/bin/env python

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ======================================================================

"""
Script to run classical molecular dynamics for a crack slab,
incrementing the load in small steps until fracture starts.

James Kermode <james.kermode@kcl.ac.uk>
August 2013
"""

import numpy as np

import ase.io
import ase.io.extxyz
import ase.units as units

from ase.constraints import FixAtoms

from matscipy.fracture_mechanics.crack import (get_strain,
                                               get_energy_release_rate,
                                               ConstantStrainRate,
                                               find_tip_stress_field,
                                               thin_strip_displacement_y)

from ase.optimize import FIRE
import sys
sys.path.insert(0, '.')
import params
import time
from quippy.atoms import Atoms as qpAtoms

# ********** Read input file ************

print 'Loading atoms from file "crack.xyz"'
atoms = ase.io.read('crack.xyz', format='extxyz')
atoms.info['is_cracked'] = False

orig_height = atoms.info['OrigHeight']
orig_crack_pos = [0., 0., 0.]

# ***** Setup constraints *******

top = atoms.positions[:, 1].max()
bottom = atoms.positions[:, 1].min()
left = atoms.positions[:, 0].min()
right = atoms.positions[:, 0].max()

# fix atoms in the top and bottom rows
if 'groups' in atoms.arrays:
    fixed_mask = atoms.get_array('groups') == 0
else:
    fixed_mask = ((abs(atoms.positions[:, 1] - top) < 1.0) |
                  (abs(atoms.positions[:, 1] - bottom) < 1.0))
fix_atoms = FixAtoms(mask=fixed_mask)

print('Fixed atoms: %d\n' % fixed_mask.sum())

# Increase epsilon_yy applied to all atoms at constant strain rate

strain_atoms = ConstantStrainRate(orig_height,
                                  params.strain_rate*params.timestep)

atoms.set_constraint([fix_atoms, strain_atoms])
atoms.set_calculator(params.calc)


# Save frames to the trajectory every `traj_interval` time steps
traj_file = open('traj-m-6r.xyz', 'w')
minimiser = FIRE(atoms, restart='hess.traj')

def trajectory_write():
    ase.io.extxyz.write_xyz(traj_file, atoms)

# swap is an array of atom indices that have to be sent in opposite direction wrt
# what thin_strip_displacement_y would do.
swap = np.loadtxt('swap_topbottom_atoms.csv')
for i in range(1000):

    minimiser.run(fmax=params.relax_fmax)
    trajectory_write()
    # Find initial position of crack tip
    #crack_pos = find_tip_stress_field(crack_slab, calc=params.calc)
    #print 'Found crack tip at position %s' % crack_pos
    
    atoms.info['strain'] = get_strain(atoms)
    strain = atoms.info['strain']
    print("Strain: %f" % strain)
    # update atomic positions
    displacement = thin_strip_displacement_y(
        atoms.positions[:, 0],
        atoms.positions[:, 1],
        params.strain_rate * params.timestep,
        left + params.crack_seed_length,
        left + params.crack_seed_length +
        params.strain_ramp_length)
    displacement[swap] = -displacement[swap]
    atoms.positions[:, 1] += displacement
