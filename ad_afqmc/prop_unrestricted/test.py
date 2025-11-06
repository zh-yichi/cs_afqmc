from pyscf import gto, scf, cc
import numpy as np

a = 2 # bond length in a cluster
d = 10 # distance between each cluster
na = 2  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
elmt = 'O'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="ccpvdz",spin=0*nc, unit='bohr', verbose=4)
mol.build()

mf = scf.UHF(mol)
e = mf.kernel()

nfrozen = 2

mycc = cc.CCSD(mf,frozen=nfrozen)
mycc.kernel()

options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 5,
            'n_blocks': 10,
            'n_walkers': 200,
            'seed': 2,
            'walker_type': 'uhf',
            'trial': 'uccsd_pt2', # ccsd_pt,ccsd_pt_ad,ccsd_pt2_ad, uccsd_pt, uccsd_pt_ad, uccsd_pt2_ad
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True,
            }

from jax import config
config.update("jax_enable_x64", True)

from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
config.setup_jax()
# config.afqmc_config = {"use_gpu": True}

from ad_afqmc import pyscf_interface, run_afqmc
from ad_afqmc.prop_unrestricted import prop_unrestricted
prop_unrestricted.prep_afqmc(mycc,options,chol_cut=1e-5)
prop_unrestricted.run_afqmc(options,nproc=5)
# pyscf_interface.prep_afqmc(mycc,options,chol_cut=1e-6)
# run_afqmc.run_afqmc(options)
