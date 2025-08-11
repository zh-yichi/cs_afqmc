from pyscf import gto, scf, cc
from ad_afqmc.lno_ccsd import lno_ccsd
import os
# import numpy as np

# T shape H2 dimer
d = 3
atoms = f'''
H     0.000     0.000    -0.370
H     0.000     0.000     0.370
H     {d}       0.000     0.000
H     {0.74+d}  0.000     0.000
'''

mol = gto.M(atom=atoms, basis="ccpvdz", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()


options = {'n_eql': 4,
           'n_prop_steps': 10,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 20,
            'n_walkers': 10,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'ad_mode':None,
            'use_gpu': False,
            }

threshs = [1e-5,1e-6]
for i,thresh in enumerate(threshs):
    lno_ccsd.run_lno_ccsd_afqmc(mycc,thresh,[],options,nproc=5)
    os.system(f'mv results.out results.out{i+1}')

lno_ccsd.sum_results(len(threshs))