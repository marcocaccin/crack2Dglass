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
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from quippy.dynamicalsystem import Dynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from matscipy.fracture_mechanics.crack import (get_strain,
                                               get_energy_release_rate,
                                               ConstantStrainRate,
                                               find_tip_stress_field)
import sys
sys.path.insert(0, '.')
import params


def get_temperature(atoms):
    KinEng = atoms.get_kinetic_energy()
    n_moving = np.logical_not(atoms.get_array('fixed_mask')).sum()
    return KinEng / (1.5 * units.kB * n_moving)

# ********** Read input file ************

print 'Loading atoms from file "crack.xyz"'
atoms = ase.io.read('cracking-6.xyz', format='extxyz')
atoms.info['is_cracked'] = False

orig_height = atoms.info['OrigHeight']
#orig_crack_pos = atoms.info['CrackPos'].copy()
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

params.strain_rate = 2.0e-6 / units.fs
strain_atoms = ConstantStrainRate(orig_height,
                                  params.strain_rate*params.timestep)

atoms.set_constraint([fix_atoms, strain_atoms])

atoms.set_calculator(params.calc)

# ********* Setup and run MD ***********

# Set the initial temperature to 2*simT: it will then equilibrate to
# simT, by the virial theorem
MaxwellBoltzmannDistribution(atoms, 2.0*params.sim_T)
p = atoms.get_momenta()
p[np.where(fixed_mask)] = 0
atoms.set_momenta(p)

# Initialise the dynamical system
dynamics = VelocityVerlet(atoms, params.timestep)
# dynamics = Langevin(atoms, params.timestep, params.sim_T * units.kB, 1e-4, fixcm=True)

# Print some information every time step
def printstatus():
    if dynamics.nsteps == 1:
        print """
State      Time/fs    Temp/K     Strain      G/(J/m^2)  CrackPos/A D(CrackPos)/A
---------------------------------------------------------------------------------"""

    log_format = ('%(label)-4s%(time)12.1f%(temperature)12.6f'+
                  '%(strain)12.4f%(crack_pos_x)12.2f    (%(d_crack_pos_x)+5.2f)')

    atoms.info['label'] = 'D'                  # Label for the status line
    atoms.info['time'] = dynamics.get_time()/units.fs
    atoms.info['temperature'] = get_temperature(atoms)
    atoms.info['strain'] = get_strain(atoms)
    # FIXME no local virial in TS, need another way to track the crack
    atoms.info['crack_pos_x'] = 0.
    atoms.info['d_crack_pos_x'] = 0.

    print log_format % atoms.info


dynamics.attach(printstatus)

# Check if the crack has advanced, and stop incrementing the strain if it has
def check_if_cracked(atoms):
    #crack_pos = find_tip_stress_field(atoms)
    # FIXME TS has no local virial
    crack_pos = [0.0, 0.0, 0.0]

    # stop straining if crack has advanced more than tip_move_tol
    if (not atoms.info['is_cracked'] and
        (crack_pos[0] - orig_crack_pos[0]) > params.tip_move_tol):
        atoms.info['is_cracked'] = True
        del atoms.constraints[atoms.constraints.index(strain_atoms)]


dynamics.attach(check_if_cracked, 1, atoms)

# Save frames to the trajectory every `traj_interval` time steps
traj_file = open('traj_verlet_6R.xyz', 'w')
trajectory_write = lambda: ase.io.extxyz.write_xyz(traj_file, atoms)
dynamics.attach(trajectory_write, params.traj_interval)

try:
    # Start running!
    dynamics.run(params.nsteps)
finally:
    traj_file.close()
