import os

import numpy as np

from ase.atoms import Atoms
from ase.io import read, write
from ase.calculators.neighborlist import NeighborList
from ase.units import GPa, J, m, kB, fs, Ang

vacuum = 15.
skin = 5.

eqm_bond_lengths = {(8, 14): 1.60,
                    (1, 8):  1.10}

crack_direction = [1, 0, 0]
cleavage_plane = [0, 1, 0]
crack_front = [0, 0, 1]

initial_strain = 0.1
crack_seed_length = 16.0
strain_ramp_length = 26.0


sim_T = 300.0*kB           # Simulation temperature
nsteps = 10000             # Total number of timesteps to run for
timestep = 0.5*fs          # Timestep (NB: time base units are not fs!)
cutoff_skin = 2.0*Ang      # Amount by which potential cutoff is increased
                                 # for neighbour calculations
tip_move_tol = 10.0              # Distance tip has to move before crack
                                 # is taken to be running
strain_rate = 1e-5 * (1/fs)  # Strain rate
traj_file = 'traj.xyz'            # Trajectory output file (NetCDF format)
traj_interval = 5               # Number of time steps between
                                 # writing output frames

relax_slab = True
relax_fmax = 0.05


a = read('SiO2_bilayer_TSfullmin.xyz', format='extxyz')
# a = read('../Crystal/2dcryst.xyz', format='extxyz')
# calculator
from quippy import Potential
calc = Potential('IP TS', param_filename='/Users/marcocaccin/OneDrive/2DGlass/ts_params.xml')
a.set_calculator(calc)

a.get_potential_energy() # obtain reference dipole moments

# fix atoms near outer boundaries
r = a.get_positions()
minx = r[:, 0].min() + skin
maxx = r[:, 0].max() - skin
miny = r[:, 1].min() + skin
maxy = r[:, 1].max() - skin
g = np.where(
    np.logical_or(
        np.logical_or(
            np.logical_or(
                r[:, 0] < minx, r[:, 0] > maxx),
            r[:, 1] < miny),
        r[:, 1] > maxy),
    np.zeros(len(a), dtype=int),
    np.ones(len(a), dtype=int))
a.set_array('groups', g)

# zero dipole moments on outer boundaries
# a.set_array('fixdip', np.logical_not(g))
# a.set_array('dipoles', calc.atoms.arrays['dipoles'])
# a.arrays['dipoles'][g==0, :] = 0.0
write('cryst.xyz', a, format='extxyz')

unit_slab = a.copy()

# pick an atom roughly in the centre of slab
r = a.get_positions()
centre = np.array([
    r[:,0].min() + 0.3 * (r[:,0].max() - r[:,0].min()),
    r[:,1].min() + 0.5 * (r[:,1].max() - r[:,1].min()),
    r[:,2].min() + 0.5 * (r[:,2].max() - r[:,2].min())
    ])
central_atom = np.argsort(map(np.linalg.norm, r - centre))[0]

surface = unit_slab.copy()
cutting_plane = surface.positions[central_atom, 1]
upper = surface.positions[:,1] > cutting_plane
lower = surface.positions[:,1] <= cutting_plane
surface.positions[upper, 1] += vacuum/2.0
surface.positions[lower, 1] -= vacuum/2.0
surface.cell[1,1] += vacuum

write('surface.xyz', surface, format='extxyz')
