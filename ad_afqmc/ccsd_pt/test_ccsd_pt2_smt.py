from pyscf import gto, scf, cc

a = 1.05835 # bond length in a cluster
d = 10 # distance between each cluster
na = 2  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
elmt = 'C'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g",spin = 0, verbose=4)
mol.build()

mf = scf.RHF(mol)
e = mf.kernel()

nfrozen = 2

# from pyscf.cc import uccsd
from ad_afqmc.ccsd_pt import ccsd_pt
mycc = cc.CCSD(mf,frozen=nfrozen)
# uccsd.UCCSD.update_amps = ccsd_pt.update_amps
mycc.kernel()

# mycc.t1 = 5*mycc.t1
# eris = mycc.ao2mo(mycc.mo_coeff)
# eccsd = mycc.energy(mycc.t1, mycc.t2, eris)
# print('ccsd energy with t1 t2: ', mf.e_tot+eccsd)

options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 5,
            'n_blocks': 10,
            'n_walkers': 20,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'ccsd_pt2_ad_smt', # ccsd_pt,ccsd_pt_ad,ccsd_pt2_ad, ucisd
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc import pyscf_interface, run_afqmc
from ad_afqmc.ccsd_pt import sample_ccsd_pt2_smt, ccsd_pt
pyscf_interface.prep_afqmc(mycc,options,chol_cut=1e-7)
sample_ccsd_pt2_smt.run_afqmc_ccsd_pt(options,nproc=8)
