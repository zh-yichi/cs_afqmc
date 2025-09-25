from pyscf import gto, scf, cc
from ad_afqmc.lno_ccsd import lno_ccsd, lno_maker
import os
# import numpy as np

# T shape H2 dimer
# d = 3
# atoms = f'''
# H     0.000     0.000    -0.370
# H     0.000     0.000     0.370
# H     {d}       0.000     0.000
# H     {0.74+d}  0.000     0.000
# '''

a = 1. # 2aB
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

options = {'n_eql': 5,
           'n_prop_steps': 30,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 5,
            'n_walkers': 1,
            'seed': 98,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'ad_mode':None,
            'use_gpu': False,
            }

# u = lno_maker.thouless_trans(mycc.t1)
# mo_t = mf.mo_coeff @ u
# mf.mo_coeff = mo_t

import numpy as np
t2 = mycc.t2
t2_0 = np.zeros(t2.shape)
mycc.t2 = t2_0
from ad_afqmc import pyscf_interface, run_afqmc
pyscf_interface.prep_afqmc(mycc,chol_cut=1e-7)

from mpi4py import MPI
MPI.Finalize()
run_afqmc.run_afqmc(options=options, nproc=5)