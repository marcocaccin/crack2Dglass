#! /usr/bin/env python

import numpy as np
import ase.io
import ase
import os
import sys
sys.path.insert(0, '.')
import shutil
import quippy
from quippy.clusters import (HYBRID_ACTIVE_MARK, HYBRID_NO_MARK, HYBRID_BUFFER_MARK,
                                                          create_hybrid_weights,
                                                          create_cluster_simple)
from quippy import Dictionary
from params import initial_strain as strain
from some_tools import relax_structure

##### PARAMETERS TO CHANGE #####
buffer_hops = 10
deltay_tip = 1.


##### OTHER STUFF, LEAVE UNCHANGED UNLESS NECESSARY #####
cluster_vacuum  = 12.0
crack_slab = ase.io.read('crack.xyz', format='extxyz')

ring_size = int(raw_input('Size of ring at crack tip:'))
pre_optim = True
try:
    pre_optim = (sys.argv[1] in [' ', 't', 'T', 'True', 'true', 'y', 'yes'])
except:
    print("") # it's alright
if pre_optim:
    crack_slab = relax_structure(crack_slab)
    
folder = 'ring%1d_strain%.03f' % (ring_size, strain)
try:
    os.mkdir(folder)
except:
    print("Folder already existing. Overwrite?")
    sig = raw_input()
    if sig not in ['y', 'yes']:
        exit

core_ring = np.loadtxt('tip_4ring.csv', dtype='int')
try:
    xy_ring = np.loadtxt('xy_ring.csv', dtype='int')
    temp = set(xy_ring)
    for idx in xy_ring:
        ring_indices = np.where(((crack_slab.positions[:,:2] - crack_slab.positions[idx,:2])**2).sum(axis=1)**0.5 < 0.5)[0]
        temp = temp.union(set(ring_indices))
    xy_ring = temp
except:
    print("WARNING: Breaking ring on XY plane not specified, using very small QM buffers")
    xy_ring = set()
hybrid_mark = np.array([1 if i in set(core_ring).union(set(xy_ring)) else 0 for i in range(len(crack_slab))])
crack_slab.set_array('hybrid_mark', hybrid_mark)

# CALC_ARGS: use first to have radius cutoff, use second for a bond hop construction
# calc_args = Dictionary('little_clusters=F terminate even_electrons cluster_vacuum=12.0 cluster_calc_connect=F buffer_hops=1 transition_hops=0 randomise_buffer=F hysteretic_connect=F nneighb_only cluster_hopping_nneighb_only property_list=species:pos:hybrid_mark:index cluster_box_buffer=20.0 cluster_hopping=F keep_whole_residues=F min_images_only keep_whole_silica_tetrahedra protect_double_bonds=F force_no_fix_termination_clash=F termination_clash_factor=1.8 nneighb_different_z in_out_in=F cluster_hopping_skip_unreachable hysteretic_buffer=T hysteretic_buffer_inner_radius=7.0 hysteretic_buffer_outer_radius=9.0')
calc_args = Dictionary('little_clusters=T terminate even_electrons cluster_vacuum=12.0 cluster_calc_connect=F buffer_hops=%d transition_hops=0 randomise_buffer=F hysteretic_connect=F nneighb_only cluster_hopping_nneighb_only property_list=species:pos:hybrid_mark:index cluster_box_buffer=20.0 cluster_hopping=T keep_whole_residues=F min_images_only keep_whole_silica_tetrahedra protect_double_bonds=F force_no_fix_termination_clash=F termination_clash_factor=1.8 nneighb_different_z in_out_in=F cluster_hopping_skip_unreachable hysteretic_buffer=F hysteretic_buffer_inner_radius=7.0 hysteretic_buffer_outer_radius=9.0 cluster_same_lattice=T' % buffer_hops)
pretty = quippy.Atoms(crack_slab)
pretty.calc_connect()
cluster_args = calc_args.copy()
create_hybrid_weights_args = calc_args.copy()
if create_hybrid_weights_args['buffer_hops'] == 0:
    create_hybrid_weights_args['buffer_hops'] = 1 # FIXME disable shortcut
create_hybrid_weights_args_str = quippy.util.args_str(create_hybrid_weights_args)
create_hybrid_weights(pretty, args_str=create_hybrid_weights_args_str)
cluster = create_cluster_simple(pretty, args_str=quippy.util.args_str(cluster_args))
cluster_indices = np.array(cluster.orig_index - 1)
fix_in_dft = np.array([i for i in cluster.indices if (cluster.hybrid_mark[i] not in [1,2])])
cluster_ase = ase.Atoms(np.array(cluster.z), np.array(cluster.positions))
shift_cluster = cluster_ase.get_positions().min(axis=0) + 0.5 * cluster_vacuum
cluster_ase.positions -= shift_cluster
cluster_ase.set_cell(cluster_ase.get_positions().ptp(axis=0) + cluster_vacuum)

abc = cluster_ase.cell.diagonal()

with open('sio22d_minim_template.inp', 'r') as fff:
    lines = fff.readlines()

# Write everything to folder
os.chdir(folder)
cluster_ase.set_array('orig_index', cluster_indices)
cluster_ase.write('ase_cluster_00.xyz', format='extxyz')
cluster_ase.write('cp2k_cluster_00.xyz', format='xyz')
os.system('tail -n +3 cp2k_cluster_00.xyz > tmp && mv tmp cp2k_cluster_00.xyz')
abc.tofile('cp2k_ABC.txt', sep=" ", format="%s")
np.savetxt('cluster_shift.csv', shift_cluster)
fix_in_dft.tofile('cp2k_fix_list.txt', sep=" ", format="%s")
np.savetxt('cluster_indices_ase.csv', cluster_indices, fmt='%d')
shutil.copy('../tip_4ring.csv', 'tip_4ring.csv')
shutil.copy('../crack.xyz', 'crack_ase.xyz')
shutil.copy('../swap_topbottom_atoms.csv', 'swap_topbottom_atoms.csv')

bottom_core_ring = core_ring[:4]
top_core_ring = core_ring[4:]
displacement_cluster = np.zeros(len(cluster_indices))
displacement_slab = np.zeros(len(crack_slab))
for idx, orig_idx in enumerate(cluster_indices):
    if orig_idx in list(top_core_ring):
        displacement_cluster[idx] = deltay_tip
        displacement_slab[orig_idx - 1] =  deltay_tip
    elif orig_idx in list(bottom_core_ring):
        displacement_cluster[idx] = - deltay_tip
        displacement_slab[orig_idx - 1] = - deltay_tip

cluster_ase.positions[:,1] += displacement_cluster
crack_slab.positions[:,1] += displacement_slab
if pre_optim:
    # temp = cluster_ase.copy()
    # temp.set_cell(cluster_ase.get_cell())
    # delete = [i for i, n in enumerate(temp.get_atomic_numbers()) if n == 1]
    # keep = [i for i, n in enumerate(temp.get_atomic_numbers()) if n != 1]
    # del temp[delete]
    # fix_TS = [i-1 for i in fix_in_dft if i-1 not in delete]
    # move_mask_TS = np.array([0 if (i in fix_TS) else 1 for i in range(len(temp))])
    # temp.set_array('move_mask', move_mask_TS)
    # temp = relax_structure(temp)
    # cluster_ase.positions[keep] = temp.get_positions()
    try:
        crack_slab = ase.io.read('../crack_open.xyz', format='extxyz')
    except:
        print('relaxing slab with open bond')
        crack_slab = relax_structure(crack_slab)
        crack_slab.write('../crack_open.xyz', format='extxyz')

    pretty.positions[:,:] = crack_slab.get_positions()
    cluster10 = create_cluster_simple(pretty, args_str=quippy.util.args_str(cluster_args))
    # remap is an essential step: re-indexes the atoms in cluster10 following the order in cluster
    # without this, there is no way to do NEB or anything that needs connection between initial and final state
    remap = [np.where(idx == cluster10.orig_index)[0].item() for idx in cluster.orig_index]

    cluster_ase = ase.Atoms(np.array(cluster10.z)[remap], np.array(cluster10.positions)[remap])
    cluster_ase.positions -= shift_cluster
    cluster_ase.set_cell(np.diag(abc))
    
cluster_ase.set_array('orig_index', cluster_indices)
cluster_ase.write('ase_cluster_10.xyz', format='extxyz')
cluster_ase.write('cp2k_cluster_10.xyz', format='xyz')
os.system('tail -n +3 cp2k_cluster_10.xyz > tmp && mv tmp cp2k_cluster_10.xyz')

line1 = lines[1][:-1]
line81 = lines[81][:-1]

lines[1] = line1 + (' %s_00\n' % folder)
lines[81] = line81 + ' \'cp2k_cluster_00.xyz\'\n'
lines[85] = lines[85][:-1] + ' ' + ' '.join([str(s) for s in abc]) + '\n'
lines[164] = lines[164][:-1] + ' ' + ' '.join([str(s) for s in fix_in_dft]) + '\n'

# Write input file for minimising final image
with open('sio22d-00.inp', 'w') as fff:
    fff.writelines(lines)

lines[1] = line1 + (' %s_10\n' % folder)
lines[81] = line81 + ' \"cp2k_cluster_10.xyz\"\n'
with open('sio22d-10.inp', 'w') as fff:
    fff.writelines(lines)

# Write the NEB input file, to be used AFTER minimising the initial and final images
with open('../neb_template.inp', 'r') as fff:
    lines = fff.readlines()
lines[74] = lines[74][:-1] + ' ' + ' '.join([str(s) for s in abc]) + '\n'
lines[126] = lines[126][:-1] + ' ' + ' '.join([str(s) for s in fix_in_dft]) + '\n'
with open('neb.inp', 'w') as fff:
    fff.writelines(lines)
shutil.copy('../run_neb_cp2k.py', 'run_neb_cp2k.py')

# HERE FOLLOWS DEV STUFF, NO USE YET
# # import silayergraph.py functions
# indices = np.where(a.get_atomic_numbers() == 14)[0]
# # asi = a[indices]
# # cutoff = 3.8
# # graph = atoms_to_nxgraph(asi, cutoff)
# # all_cycles = minimal_cycles(graph, cutoff=9)
# all_cycles = pkl.load('slab_cycles.pkl')
# core_cycle_index = [i for i, c in enumerate(all_cycles) if set(core_cycle) == set(list(indices[list(c)]))][0]
