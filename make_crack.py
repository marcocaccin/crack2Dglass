#! /usr/bin/env python

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
Script to generate a crack slab, and apply initial strain ramp

James Kermode <james.kermode@kcl.ac.uk>
August 2014
"""

import numpy as np

import ase.io
import ase.units as units
from matscipy.fracture_mechanics.crack import thin_strip_displacement_y
from some_tools import relax_structure, shrink_cell

import sys
sys.path.insert(0, '.')
import params

swap = np.loadtxt('swap_topbottom_atoms.csv', dtype='int')
a = params.a.copy()
a.set_calculator(params.calc)

directions = [params.crack_direction,
              params.cleavage_plane,
              params.crack_front]

# now, we build system aligned with requested crystallographic orientation
unit_slab = params.unit_slab

print('Unit slab with %d atoms per unit cell:' % len(unit_slab))
print(unit_slab.cell)
print('')

pos_0 = np.array([-21.5, -2.9, 31.])
pos_1 = np.array([-21.4, -1.38, 30.6])
r = unit_slab.get_positions()
atom_0, atom_1 = [np.argsort([np.linalg.norm(r_i - pos) for r_i in r])[0]
                  for pos in [pos_0, pos_1]]
# atom_0, atom_1 = 0, 1
# center vertically half way along the vertical bond between atoms 0 and 1
unit_slab.positions[:, 1] += (unit_slab.positions[atom_1, 1] -
                              unit_slab.positions[atom_0, 1]) / 2.0

# map positions back into unit cell
unit_slab.set_scaled_positions(unit_slab.get_scaled_positions())

if hasattr(params, 'surface'):
    surface = params.surface
else:
    # Make a surface unit cell by repllcating and adding some vaccum along y
    surface = unit_slab * [1, params.surf_ny, 1]
    surface.center(params.vacuum, axis=1)


# ********** Surface energy ************

# Calculate surface energy per unit area
surface.set_calculator(params.calc)

if hasattr(params, 'relax_bulk') and params.relax_bulk:
    print('Minimising surface unit cell...')
    surface = relax_structure(surface)

E_surf = surface.get_potential_energy()
E_per_atom_bulk = a.get_potential_energy() / len(a)

# volume is not get_volume() because it's not a solid 3D system!
volume = np.prod([np.ptp(vect) for vect in a.get_positions().T])
area = volume / np.ptp(a.get_positions()[:,1])
gamma = ((E_surf - E_per_atom_bulk * len(surface)) /
         (2.0 * area))

print('Surface energy of %s surface %.4f J/m^2\n' %
      (params.cleavage_plane, gamma / (units.J / units.m ** 2)))

# ***** Setup crack slab supercell *****
crack_slab = unit_slab

# open up the cell along x and y by introducing some vacuum
shrink_cell(crack_slab)
crack_slab.center(params.vacuum, axis=0)
crack_slab.center(params.vacuum, axis=1)

# centre the slab on the origin
crack_slab.positions[:, 0] -= crack_slab.positions[:, 0].mean()
crack_slab.positions[:, 1] -= crack_slab.positions[:, 1].mean()
crack_slab.positions[:, 1] += 0.5

top = crack_slab.positions[:, 1].max()
bottom = crack_slab.positions[:, 1].min()
left = crack_slab.positions[:, 0].min()
right = crack_slab.positions[:, 0].max()

orig_width = right - left
orig_height = top - bottom

print(('Made slab with %d atoms, original width and height: %.1f x %.1f A^2' %
       (len(crack_slab), orig_width, orig_height)))


# ****** Apply initial strain ramp *****

strain = params.initial_strain

displacement = thin_strip_displacement_y(
    crack_slab.positions[:, 0],
    crack_slab.positions[:, 1],
    strain,
    left + params.crack_seed_length,
    left + params.crack_seed_length +
    params.strain_ramp_length)

displacement[swap] = -displacement[swap]
crack_slab.positions[:, 1] += displacement
# cleanup_crack_tip(crack_slab)

# fix atoms in the top and bottom rows
if 'groups' in crack_slab.arrays:
    fixed_mask = crack_slab.get_array('groups') == 0
else:
    fixed_mask = ((abs(crack_slab.positions[:, 1] - top) < 1.0) |
                  (abs(crack_slab.positions[:, 1] - bottom) < 1.0))
print('Fixed atoms: %d\n' % fixed_mask.sum())

# Save all calculated materials properties inside the Atoms object
crack_slab.info['nneightol'] = 1.3 # nearest neighbour tolerance
crack_slab.info['SurfaceEnergy'] = gamma
crack_slab.info['OrigWidth'] = orig_width
crack_slab.info['OrigHeight'] = orig_height
crack_slab.info['CrackDirection'] = params.crack_direction
crack_slab.info['CleavagePlane'] = params.cleavage_plane
crack_slab.info['CrackFront'] = params.crack_front
crack_slab.info['cell_origin'] = -np.diag(crack_slab.cell)/2.0

crack_slab.set_array('fixed_mask', fixed_mask)
ase.io.write('slab.xyz', crack_slab, format='extxyz')

print('Applied initial load: strain=%.4f' % strain)


# ***** Relaxation of crack slab  *****

# optionally, relax the slab, keeping top and bottom rows fixed
if params.relax_slab:
    ase.io.write('crack.xyz', crack_slab, format='extxyz')
    crack_slab = relax_structure(crack_slab)

crack_slab.info['strain'] = strain
crack_slab.info['is_cracked'] = False

# ******** Save output file **********

# Save results in extended XYZ format, including extra properties and info
print('Writing crack slab to file "crack.xyz"')
ase.io.write('crack.xyz', crack_slab, format='extxyz')
