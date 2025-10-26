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

mf = scf.RHF(mol)
e = mf.kernel()
# print(mf.mo_coeff)
nfrozen = 0

# s1e = mf.get_ovlp()
# olp = mf.mo_coeff[0].T @ s1e @ mf.mo_coeff[1]
# sign = np.array(np.sign(olp.diagonal()), dtype=int)
# print('<A|B> sign: ', sign)
# if -1 not in sign:
    # sign[1] = -1
#    mf.mo_coeff[1][:,1] = -mf.mo_coeff[1][:,1]
# olp = mf.mo_coeff[0].T @ s1e @ mf.mo_coeff[1]
# sign = np.array(np.sign(olp.diagonal()), dtype=int)
# mf.mo_coeff[1] = np.einsum('ij,j->ij',mf.mo_coeff[1],sign)
# olp = mf.mo_coeff[0].T @ s1e @ mf.mo_coeff[1]
# sign = np.array(np.sign(olp.diagonal()), dtype=int)
# print('new <A|B> sign: ', sign)

# from pyscf.cc import uccsd
# from ad_afqmc.ccsd_pt import ccsd_pt
mycc = cc.CCSD(mf,frozen=nfrozen)
# uccsd.UCCSD.update_amps = ccsd_pt.update_amps
mycc.kernel()
# print(mf.mo_coeff)
# mycc.t1 = (10*mycc.t1[0],10*mycc.t1[1])
# mycc.t1 = mycc.t1*10
# eris = mycc.ao2mo(mycc.mo_coeff)
# eccs = mycc.energy(mycc.t1, (0*mycc.t2[0],0*mycc.t2[1],0*mycc.t2[2]), eris)
# # eccs = mycc.energy(mycc.t1, mycc.t2*0)
# print('ccs energy with 10*t1 0*t2: ', mf.e_tot+eccs)
# eccsd = mycc.energy(mycc.t1, mycc.t2, eris)
# print('ccsd energy with 10*t1 t2: ', mf.e_tot+eccsd)

options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 5,
            'n_blocks': 10,
            'n_walkers': 30,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'ccsd_pt2_ad', # ccsd_pt,ccsd_pt_ad,ccsd_pt2_ad, uccsd_pt, uccsd_pt_ad, uccsd_pt2_ad
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc import pyscf_interface
from ad_afqmc.ccsd_pt import sample_ccsd_pt2
pyscf_interface.prep_afqmc(mycc,options,chol_cut=1e-5)
sample_ccsd_pt2.run_afqmc(options,nproc=5)
