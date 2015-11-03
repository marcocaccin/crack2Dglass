import sys
import ase
from ase.io import read
# from quippy.atoms import Atoms
import numpy as np

filename = sys.argv[1]
wet = read(filename + '.xyz')
h2omol = ase.atoms.Atoms(numbers=[8,1,1],
                         positions = np.array([[19.41412388, 26.61580360, 3.31181618],
                                               [20.01309382, 27.38313272, 3.31181618],
                                               [20.01309382, 25.84847448, 3.31181618]]))
# [[-18.78806296, 1.43344965, -1.60279923],
# [-18.18909302, -0.66612053, -1.60279923],
# [-18.18909302, -2.20077877, -1.60279923]]))


cell = np.diag(wet.get_positions().ptp(axis=0) + 12.)
wet.set_cell(cell)
wet.set_pbc([True]*3)
h2omol.positions[:,:] += wet.get_positions().mean(axis=0) - h2omol.get_positions().mean(axis=0)
wet.extend(h2omol)

wet.write('%s_wet.xyz' % filename, format='extxyz')
