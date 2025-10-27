from pyscf import gto, scf, cc
import numpy as np

a = 2 # bond length in a cluster
d = 10 # distance between each cluster
na = 4  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
elmt = 'H'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g",spin=0*nc, unit='bohr', verbose=4)
mol.build()

mf = scf.UHF(mol)
e = mf.kernel()

# mf.mo_coeff[1] += 1e-4
# mf.make_rdm1(mf.mo_coeff,mf.mo_occ)
# mf.max_cycle = -1
# mf.kernel()

nfrozen = 0

mycc = cc.CCSD(mf,frozen=nfrozen)
mycc.kernel()

options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 10,
            'n_blocks': 10,
            'n_walkers': 30,
            'seed': 2,
            'walker_type': 'uhf',
            'trial': 'uhf', # ccsd_pt,ccsd_pt_ad,ccsd_pt2_ad, uccsd_pt, uccsd_pt_ad, uccsd_pt2_ad
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc import pyscf_interface, run_afqmc
from ad_afqmc.ccsd_pt import prop_unrestricted
prop_unrestricted.prep_afqmc(mycc,options,chol_cut=1e-5)
prop_unrestricted.run_afqmc(options,nproc=5)
# pyscf_interface.prep_afqmc(mf,options,chol_cut=1e-5)
# run_afqmc.run_afqmc(options,nproc=5)
