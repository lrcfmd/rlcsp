#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:41:17 2020

The input file to run fuse with the Uniform policy for Sr4Ti3O10

@author: Chris Collins
"""
from fuse106 import *
import sys
import time

#### composition information
#composition = {'Y':[2],'Ti':[2],'O':[7]} #composition in non-charged format
composition = {'Sr':[4,+2],'Ti':[3,+4],'O':[10,-2]} #composition in ionic format

#### set search routine 1 = BH, 2 = GA
search = 1

### Genetic algo inputs (running on defaults)

### Basin hopping inputs (running on defaults)

### general inputs
rmax=1000000 # the number of steps since the global minimum was located before ending
#the calculation
iterations=2000 # number of structures to compute in this run
restart=False # restart a previous run?
initial_gen=100 # size of the initial population
search_gen=1 # size of each generation in to be used by the search routine
max_atoms=50 # global limit on the maximum number of atoms
imax_atoms=50 # maximum number of atoms to be used in the initial population

### gulp inputs
ctype='gulp' # flag to pass to ase inorder to use gulp as the energy minimiser
kwds=['opti conj conp c6','opti lbfgs conp c6'] # keywords for the gulp input,
#multiple strings indicate running GULP in multiple stages
gulp_opts=[
['\nlibrary lib2.lib\ndump temp.res\ntime 15 minuets\nmaxcyc 50\nstepmax 0.1\ntime 5 minutes'],
['\nlibrary lib2.lib\ndump temp.res\ntime 15 minutes\nmaxcyc 1500\nlbfgs_order 5000\ntime 5 minutes'],
]	# options for gulp, must include one line per set of inputs in "kwds"
lib='lib2.lib' # library file for interatomic potentials
shel=['']	# species using shells in gulp

# output options (running on defaults)

# run fuse

if len(sys.argv) > 1:
    dir_num = sys.argv[1]
else:
    dir_num = '1'

run_fuse(composition=composition,search=search,rmax=rmax,
	iterations=iterations,restart=restart,initial_gen=initial_gen,
	search_gen=search_gen,max_atoms=max_atoms,imax_atoms=imax_atoms,ctype=ctype,
	kwds=kwds,gulp_opts=gulp_opts,lib=lib,shel=shel)

sys.exit()
