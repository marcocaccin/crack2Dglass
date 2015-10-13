run make_crack_guess.py -> get cracked specimen, unrelaxed and with wrong ordering of O above and below surfaces.
Inspect crack_guess.xyz and select by hand the atoms that we want to send in opposite direction wrt current guess (Ovito perspective is good).
Edit 'swap_topbottom_atoms.csv' with atom indices to be swapped on the other side of the crack (zero-based)
run make_crack.py and obtain TS minimised structure in crack.xyz
Inspect the file 'crack.xyz' and identify atoms in the 4-ring at the crack tip. Write them in 'tip_4ring.csv' (zero-based): first 4 atoms go above crack, last 4 go below
Run 'make_cluster_from_crack.py': this will create a subfolder containing all the useful files for running cp2k minimisation of the carved clusters and also everything necessary to put back the clusters into the main slab.
