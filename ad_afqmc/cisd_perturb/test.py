from pyscf import gto, scf, cc
import numpy as np
from jax import numpy as jnp
from jax import vmap

a = 1.5 # 2aB
nH = 4
atoms = ""
for i in range(nH):
    atoms += f"H {i*a:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mol.build()

mf = scf.RHF(mol)#.density_fit()
e = mf.kernel()

mycc = cc.CCSD(mf)
e = mycc.kernel()

options = {'n_eql': 4,
           'n_prop_steps': 10,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 10,
            'n_walkers': 2,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc import pyscf_interface, mpi_jax, wavefunctions
from ad_afqmc.cisd_perturb import sample_pt
pyscf_interface.prep_afqmc(mycc,chol_cut=1e-7)

sample_pt.run_afqmc_cisd_pt(options,nproc=5)
