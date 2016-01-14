"""
Collection of functions to manipulate 2D glass systems.
"""


import numpy as np

from ase.constraints import FixAtoms
from ase.optimize import FIRE, LBFGS
import ase.io
import ase.units as units
from ase import Atom

from matscipy.neighbours import neighbour_list
import numpy.linalg as LA
from quippy import Potential
from quippy import Atoms as QAtoms
from ase.atoms import Atoms as AAtoms


calc = Potential('IP TS', param_filename='/Users/marcocaccin/OneDrive/2DGlass/ts_params.xml')


def relax_structure(atoms, relax_fmax=0.05, traj_interval=None):
    """
    Relax atoms object. If it contains a 'fixed_mask' array, then run constrained relaxation
    """
    arel = atoms.copy()
    try:
        fix_mask = atoms.get_array('fixed_mask')
        print("Running relaxation with %d constrained atoms" % fix_mask.sum())
        const = FixAtoms(mask=fix_mask)
        arel.set_constraint(const)
    except:
        print("No constraints specified, running relaxation")
    arel.set_calculator(calc)
    opt = FIRE(arel) # LBFGS(arel) #
    if traj_interval is not None:
        from quippy.io import AtomsWriter
        out = AtomsWriter("traj-relax_structure.xyz")
        trajectory_write = lambda: out.write(QAtoms(arel, charge=None))
        opt.attach(trajectory_write, interval=traj_interval)
    opt.run(fmax=relax_fmax)
    return arel


def shrink_cell(atoms):
    """
    Shrink Atoms cell to match min-max in every direction
    """
    r = atoms.get_positions()
    mins = [pos.min() for pos in r.T]
    maxs = [pos.max() for pos in r.T]
    atoms.set_cell([b - a for a, b in zip(mins, maxs)])
    return


def relax_bare_cluster(cluster, relax_fmax=0.05):
    c0 = cluster.copy()
    orig_index = c0.get_array('orig_index')
    fix_list = np.loadtxt('cp2k_fix_list.txt', dtype='int') - 1 
    fix_orig_index = [orig_index[i] for i in fix_list]
    hydrogens = np.where(c0.get_atomic_numbers() == 1)[0]
    del c0[hydrogens]

    new_fix_list = [i for i, j in enumerate(c0.get_array('orig_index')) if j in fix_orig_index]
    pot = calc
    c0.set_calculator(pot)
    c0.get_potential_energy()
    c0.set_array('fixdip', np.array([True if i in new_fix_list else False for i in range(len(c0))]))
    c0.set_array('dipoles', np.zeros((len(c0),3))) # pot.atoms.arrays['dipoles'])
    # c0.arrays['dipoles'][new_fix_list, :] = 0.0

    const = FixAtoms(indices=new_fix_list)
    c0.set_constraint(const)
    opt = LBFGS(c0)
    opt.run(fmax=relax_fmax)
    return c0


def tetrahedron_4th_position(tetrahedron_center, tripod_positions):
    """
    Return position of 4th element of a tetrahedron-like object given its center and 3 corners.
    """
    assert np.asarray(tripod_positions).shape == (3,3)
    assert np.asarray(tetrahedron_center).shape == (3,)

    v1, v2 = tripod_positions[1] - tripod_positions[0], tripod_positions[2] - tripod_positions[0]
    normal = np.cross(v1, v2)
    normal /= LA.norm(normal)
    which_side = np.dot(normal, tetrahedron_center - tripod_positions[0]) >= 0.
    mean_distance = np.mean(map(LA.norm, tripod_positions - tetrahedron_center[:, None]))
    return (2*int(which_side)-1) * mean_distance*normal + tetrahedron_center


def complete_Si_tetrahedrons(atoms, cutoff=2.):
    """
    Every 3-coordinated Si atom gets an extra O atom to complete the tetrahedron.
    """
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
    Orientation of O--H bond is rather arbitrary.
    """

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


def remove_dangling_atoms(atoms, keep_cell=False, write=False):

    if not keep_cell:
        cell0 = atoms.get_cell()
        atoms.set_cell([1000]*3)
        atoms.set_pbc([True]*3)
        
    # atoms.set_cutoff(2.)
    # atoms.calc_connect()
    
    # remove dangling atoms not in loops
    dropped = 1
    idx = 0
    while dropped > 0:
        # nneighs = np.array([len(atoms.neighbours[i]) for i in atoms.indices]) # quip implementation is slow
        ni = neighbour_list('i', atoms, 2.)
        nneighs = np.array([(ni == i).sum() for i in range(len(atoms))])
        speciesSi = atoms.get_atomic_numbers() == 14
        mask = np.array([(nneigh < 3) if issilicon else (nneigh < 2) 
                         for issilicon, nneigh in zip(speciesSi, nneighs)])
        if write:
            atoms.set_array('remove', mask)
            atoms.write('rem-%03d.xyz' % idx, format='extxyz')
        if type(atoms) == QAtoms:
            atoms.remove_atoms(mask=mask)
        else:
            del atoms[np.where(mask)]
        dropped = mask.sum()
        idx +=1
    if not keep_cell:
        atoms.set_cell(cell0)
    return
    
    
def SiO2_2D_cubic(a=1.6, vacuum=12.0):

    """
    Create a cubic (parallelepiped) unit cell of a bilayer 2D silica structure.


    Inputs:
    --------
        a : float, default = 1.6
            Si--O bond length
        vacuum : float, default = 12.0
            Size of vacuum in z direction

    Returns:
    --------
        unit_cell : quippy Atoms object
            Unit cell of the specified structure.
    """

    from quippy import graphene_cubic
    
    top = graphene_cubic(1)
    top.positions[:,2] = 0.5
    top.set_chemical_symbols([14]*len(top))
    top.add_atoms(pos=np.array([[0.25, 0.25*np.sqrt(3), 0.625],
                                [0.25, 0.75*np.sqrt(3), 0.625],
                                [1.  , 0.             , 0.625],
                                [1.75, 0.75*np.sqrt(3), 0.625],
                                [1.75, 0.25*np.sqrt(3), 0.625],
                                [2.5 , 0.5*np.sqrt(3) , 0.625]]).T,
                  z=[8]*6)
    bottom = top.copy()
    bottom.positions[:,2] = - bottom.get_positions()[:,2]
    unit_cell = graphene_cubic(1)
    unit_cell.set_chemical_symbols([8]*len(unit_cell))
    unit_cell.add_atoms(top)
    unit_cell.add_atoms(bottom)
    unit_cell.set_cell(2*a*unit_cell.get_cell(), scale_atoms=True)
    unit_cell.cell[2,2] = 2.5*a + vacuum
    return unit_cell

def double_relax(atoms, relax_fmax = 0.05):
    """
    Relax first with ASE optimiser, which allows constrained minimisation but does not allow cell optimisation.
    Then relax with QUIP routine, which allows cell opt but is unconstrained.
    """
    if type(atoms) == AAtoms:
        tp = 'A'
    elif type(atoms) == QAtoms:
        tp = 'Q'
        atoms = AAtoms(atoms)
    else:
        print("Error: unknown Atoms type")
        return
    
    atoms = relax_structure(atoms, relax_fmax = relax_fmax)
    atoms = QAtoms(atoms)
    atoms.write('temp.xyz')
    calc.minim(atoms, 'cg', relax_fmax, 1000, do_pos=True, do_lat=False)
    if tp == 'A':
        atoms = AAtoms(atoms)
