import numpy as np

from ase.constraints import FixAtoms
from ase.optimize import FIRE, LBFGS
import ase.io
import ase.units as units

import sys
sys.path.insert(0, '.')
import params

def relax_structure(atoms):
    arel = atoms.copy()
    try:
        fix_mask = atoms.get_array('fixed_mask')
        print("Running relaxation with %d constrained atoms" % fix_mask.sum())
        const = FixAtoms(mask=fix_mask)
        arel.set_constraint(const)
    except:
        print("No constraints specified, running relaxation")
    arel.set_calculator(params.calc)
    opt = FIRE(arel) # LBFGS(arel) # 
    opt.run(fmax=params.relax_fmax)
    return arel


def shrink_cell(atoms):
    r = atoms.get_positions()
    mins = [pos.min() for pos in r.T]
    maxs = [pos.max() for pos in r.T]
    atoms.set_cell([b - a for a, b in zip(mins, maxs)])
    return


def relax_bare_cluster(cluster):
    c0 = cluster.copy()
    orig_index = c0.get_array('orig_index')
    fix_list = np.loadtxt('cp2k_fix_list.txt', dtype='int') - 1 
    fix_orig_index = [orig_index[i] for i in fix_list]
    hydrogens = np.where(c0.get_atomic_numbers() == 1)[0]
    del c0[hydrogens]

    new_fix_list = [i for i, j in enumerate(c0.get_array('orig_index')) if j in fix_orig_index]
    pot = params.calc
    c0.set_calculator(pot)
    c0.get_potential_energy()
    c0.set_array('fixdip', np.array([True if i in new_fix_list else False for i in range(len(c0))]))
    c0.set_array('dipoles', np.zeros((len(c0),3))) # pot.atoms.arrays['dipoles'])
    # c0.arrays['dipoles'][new_fix_list, :] = 0.0

    const = FixAtoms(indices=new_fix_list)
    c0.set_constraint(const)
    opt = LBFGS(c0)
    opt.run(fmax=params.relax_fmax)
    return c0


def tetrahedron_4th_position(tetrahedron_center, tripod_positions):
    """
    Return position of 4th element of a tetrahedron-like object given its center and 3 corners.
    """
    assert np.asarray(tripod_positions).shape == (3,3)
    assert np.asarray(tetrahedron_center).shape == (3,)

    import numpy.linalg as LA

    v1, v2 = tripod_positions[1] - tripod_positions[0], tripod_positions[2] - tripod_positions[0]
    normal = np.cross(v1, v2)
    normal /= LA.norm(normal)
    which_side = np.dot(normal, tetrahedron_center - tripod_positions[0]) >= 0.
    mean_distance = np.mean(map(LA.norm, tripod_positions - tetrahedron_center[:, None]))
    return (2*int(which_side)-1) * mean_distance*normal + tetrahedron_center


def complete_Si_tetrahedrons(atoms, cutoff=2.):
    from matscipy.neighbours import neighbour_list
    from ase import Atom

    species = atoms.get_atomic_numbers()
    ii, jj, DD, SS = neighbour_list('ijDS', atoms, cutoff)
    for i in range(len(atoms)):
        if species[i] == 14:
            neighbours = np.where(ii == i)[0]
            distances = DD[neighbours]
            if len(neighbours) == 3:
                pos4thO = tetrahedron_4th_position(np.zeros(3), distances)
                atoms.append(Atom(symbol = 'O', position = pos4thO + atoms.get_positions()[i]))
    return


def hydrogenate_Os(atoms, OHdistance=0.98, cutoff=2., mask=None):
    """
    Every Oxygen atom that sticks out undercoordinated gets a H atom.
    """
    from matscipy.neighbours import neighbour_list
    from ase import Atom
    import numpy.linalg as LA

    if mask is None:
        mask = np.ones(len(atoms))
    
    ii, jj, DD, SS = neighbour_list('ijDS', atoms, cutoff)

    species = atoms.get_atomic_numbers()
    for i, yes in zip(range(len(atoms)), mask):
        if species[i] == 8 and yes:
            neighbours = np.where(ii == i)[0]
            if len(neighbours) == 1:
                distance = DD[neighbours].flatten()
                posH = atoms.get_positions()[i] - OHdistance*distance/LA.norm(distance)
                atoms.append(Atom(symbol = 'H', position = posH))
    return
