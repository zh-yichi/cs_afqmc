from pyscf import gto, scf, cc
from ad_afqmc.lno_ccsd import lno_ccsd
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
nH = 10
atoms = ""
for i in range(nH):
    atoms += f"H {i*a:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

options = {'n_eql': 4,
           'n_prop_steps': 30,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 10,
            'n_walkers': 10,
            'seed': 98,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'ad_mode':None,
            'use_gpu': False,
            }

# threshs = [1e-4,1e-5,1e-6,1e-7]
# for i,thresh in enumerate(threshs):
#     lno_ccsd.run_lno_ccsd_afqmc(mf,thresh,[],options,nproc=5)
#     os.system(f'mv results.out results.out{i+1}')

# lno_ccsd.sum_results(len(threshs))

from ad_afqmc import pyscf_interface, run_afqmc
pyscf_interface.prep_afqmc(mycc)

from mpi4py import MPI

MPI.Finalize()
run_afqmc.run_afqmc(options=options, nproc=5)