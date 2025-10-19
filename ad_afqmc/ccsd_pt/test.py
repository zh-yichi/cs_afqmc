from pyscf import gto, scf, cc

a = 1.05835 # bond length in a cluster
d = 10 # distance between each cluster
na = 2  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
elmt = 'H'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g",spin = 0, verbose=4)
# mol.charge = 0*nc
# mol.spin = 1
mol.build()

mf = scf.UHF(mol)
e = mf.kernel()

nfrozen = 0

from pyscf.cc import uccsd
from ad_afqmc.ccsd_pt import ccsd_pt
mycc = uccsd.UCCSD(mf,frozen=nfrozen)
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
            'n_walkers': 10,
            'seed': 2,
            'walker_type': 'uhf',
            'trial': 'ucisd', # ccsd_pt,ccsd_pt_ad,ccsd_pt2_ad, ucisd
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc import pyscf_interface, run_afqmc
from ad_afqmc.ccsd_pt import sample_ccsd_pt, sample_ccsd_pt2, sample_uccsd_pt, ccsd_pt
ccsd_pt.prep_afqmc(mycc,chol_cut=1e-5)
# pyscf_interface.prep_afqmc(mycc,options,chol_cut=1e-5)
# sample_ccsd_pt.run_afqmc_ccsd_pt(options,nproc=8)
sample_uccsd_pt.run_afqmc_uccsd_pt(options,nproc=8)
# sample_ccsd_pt.run_afqmc_ccsd_pt(options,nproc=4,script='run_afqmc_ccsd_pt_test.py')
# run_afqmc.run_afqmc(options,nproc=5)
