import sys

import numpy as np

import pyximport; pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True)
from looplib import loopviz, looptools, simlef_onesided, simlef
import os, sys, glob, shelve, time

import seaborn as sns
sns.set_style('white')


# Simulation Parameters
p = {}
p['L'] = 20000                  # 20000
p['N'] = 20
p['R_OFF'] = 1.0 / 1000.0       # 1.0 / 1000.0
p['R_EXTEND'] = float(2.0)      # 2.0
p['R_SHRINK'] = float(0.1)      # 0.1

p['R_SWITCH'] = p['R_OFF'] * 10     # p['R_OFF'] * 10

p['T_MAX_LIFETIMES'] = 100.0        # 100.0
p['T_MAX'] = p['T_MAX_LIFETIMES'] / p['R_OFF']
p['N_SNAPSHOTS'] = 1000             # 200
p['PROCESS_NAME'] = b'proc'         # b'proc'

l_sites, r_sites, ts = simlef.simulate(p, verbose = True)


# Plotting Results
