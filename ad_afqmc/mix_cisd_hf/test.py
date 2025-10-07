from pyscf import gto, scf, cc
import numpy as np
from jax import numpy as jnp
from jax import vmap

a = 1 # 2aB
d = 10
nH = 12 # set as integer multiple of 4
atoms = ""
if nH == 2:
    for i in range(nH):
        atoms += f"H {i*a:.5f} 0.00000 0.00000 \n"
else:
    for n in range(nH):
        shift = ((n - n % 4) // 4) * (d-1)
        atoms += f"H {n*a + shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mol.build()

mf = scf.RHF(mol)#.density_fit()
e = mf.kernel()

mycc = cc.CCSD(mf)
e = mycc.kernel()

options = {'n_eql': 4,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 10,
            'n_blocks': 10,
            'n_walkers': 10,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc import pyscf_interface
from ad_afqmc.mix_cisd_hf import propgate_mix
from ad_afqmc.cisd_perturb import sample_pt
pyscf_interface.prep_afqmc(mycc,chol_cut=1e-7)

propgate_mix.run_mixed_prop(options,nproc=5)
# sample_pt.run_afqmc_cisd_pt(options,nproc=5)