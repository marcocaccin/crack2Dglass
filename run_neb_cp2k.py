#!/usr/bin/env python

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import sys
import os
import subprocess

from time import sleep
from bgqtools import (get_bootable_blocks, boot_blocks, block_corner_iter,
                      set_unbuffered_stdout)

set_unbuffered_stdout()

"""
script description:

run njobs cp2k instances concurrently on subpartitions. 
Input file is ('%s.inp' % jobstem) for all runs or ('%s-1.restart' % project) in case of restart. inp must be present in main directory, restart has to be in each of the job directories, as it is created by a previous cp2k run.
xyz file is different for each run and follow naming ('%s_%d.xyz' % (jobstem, job_idx)), and must be present in main directory

WARNING: in file *.inp change PROJECT to match jobstem, so that the restart file is created with the right name for successive restarts.
"""

# ESSENTIAL JOB VARIABLES SETUP. Only change stuff here

time = 360
nodes = 1024 # nodes per job
ppn = 1 # MPI tasks per node
threads = 16 # openMP threads per node
jobstem = 'neb'


# Get input files ready for the neb calculation
cwd = os.getcwd().split('/')[-1]
for idx in [0,10]:
    atname = '%s_%02d-pos-1.xyz' % (cwd,idx)
    with open(atname, 'r') as fff:
        nat = int(fff.readline())
    os.system('tail -%d %s > neb_%02d.xyz' % (nat, atname, idx))
    os.system('awk \'{print $2, $3, $4}\' neb_%02d.xyz > neb_%02donly.xyz' % (idx,idx))
    
# OTHER VARIABLES SETUP

acct = 'SiO2_Fracture'
queue = 'default'
mapping = 'ABCDET'
scratch = os.getcwd()
exe = '/home/avazquez/public/install/bin/cp2k.psmp' # '/soft/applications/cp2k/cp2k.psmp-2.5'
envargs = '--envs OMP_NUM_THREADS=%d BG_SMP_FAST_WAKEUP=YES BG_THREADLAYOUT=1 OMP_WAIT_POLICY=ACTIVE' % threads

# ACTUAL CODE
if 'COBALT_PARTSIZE' not in os.environ:
    print('Not running under control of cobalt. Launching qsub...')
    qsub_args = 'qsub -A %s -n %d -t %d -q %s --mode script --disable_preboot %s' % (acct, nodes, time, queue, sys.argv[0])
    print(qsub_args)
    os.system(qsub_args)
    sys.exit(1)

partsize = int(os.environ['COBALT_PARTSIZE'])
partition = os.environ['COBALT_PARTNAME']
jobid = int(os.environ['COBALT_JOBID'])
# blocks = get_bootable_blocks(partition, nodes)
boot_blocks([partition])
print("partition: %s" % partition)

if os.path.exists('%s-1.restart' % jobstem):
    # use restart as input
    input_file = '%s-1.restart' % jobstem
else:
    # files needed: inp and xyz
    input_file = jobstem + '.inp'
# locargs = '--block %s' % blocks[0]
runjob_args = ('runjob --np %d -p %d --block %s %s : %s %s' % (nodes*ppn, ppn, partition, envargs, exe, input_file)).split()
print(' '.join(runjob_args))

job = subprocess.Popen(runjob_args)
# wait for all background jobs to finish
job.wait()

