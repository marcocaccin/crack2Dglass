import numpy as np

from ase.constraints import FixAtoms
from ase.optimize import FIRE
import ase.io
import ase.units as units

import sys
sys.path.insert(0, '.')
import params

def relax_structure(atoms):
    fix_mask = atoms.get_array('fixed_mask')
    print("Running relaxation with %d constrained atoms" % fix_mask.sum())
    const = FixAtoms(mask=fix_mask)
    arel = atoms.copy()
    arel.set_constraint(const)
    arel.set_calculator(params.calc)
    opt = FIRE(arel)
    opt.run(fmax=params.relax_fmax)
    return arel


def shrink_cell(atoms):
    r = atoms.get_positions()
    mins = [pos.min() for pos in r.T]
    maxs = [pos.max() for pos in r.T]
    atoms.set_cell([b - a for a, b in zip(mins, maxs)])
    return
