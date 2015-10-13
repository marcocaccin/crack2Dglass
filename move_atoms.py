import ase.io
import numpy as np
import os
broken_bond = False
if broken_bond:
    brk = '10'
else:
    brk = '00'
file1 = '/Users/marcocaccin/Desktop/ring10_strain0.100_%s-pos-1.xyz' % brk
newfld = 'ring68_strain0.100'
oldfld = 'ring610_strain0.100'

displacement1to2 = np.loadtxt('%s/cluster_shift.csv' % newfld) - np.loadtxt('%s/cluster_shift.csv' % oldfld)

c1 = ase.io.read('%s/%s_%s-pos-1.xyz' % (oldfld, oldfld, brk), index=-1)
c2 = ase.io.read('%s/ase_cluster_%s.xyz' % (newfld, brk))

cluster_idx1 = np.loadtxt('%s/cluster_indices_ase.csv' % oldfld, dtype='int')
cluster_idx2 = np.loadtxt('%s/cluster_indices_ase.csv' % newfld, dtype='int')

cluster_idx1H = cluster_idx1[c1.get_atomic_numbers() != 1]
cluster_idx2H = cluster_idx2[c2.get_atomic_numbers() != 1]
common_idx = set(cluster_idx1H).intersection(set(cluster_idx2H))

for j2, i2 in enumerate(cluster_idx2): # loop over each atom in c2... 
    if i2 in common_idx: # if this atom is in common between c1 and c2, and is not a terminating H in either structure...
        j1 = np.where(i2 == cluster_idx1)[0].item() # get index of atom in configuration c1 
        deltar = c1.get_positions()[j1] - c2.get_positions()[j2] - displacement1to2
        print(i2, deltar)
        c2.positions[j2] += deltar # replace position of atom with the one contained in c1, adjusting possible cell shifts
        
ase.io.write('%s/ase_cluster_%s_updated.xyz' % (newfld, brk), c2, format='extxyz')
ase.io.write('tmp.xyz', c2)
os.system('tail -n +3 tmp.xyz > tmp && mv tmp %s/cp2k_cluster_%s_updated.xyz' % (newfld, brk))
