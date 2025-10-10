from pyscf import gto, scf, cc

a = 1.05835
d = 10
nH = 2 # set as integer multiple of 2
atoms = ""
for n in range(nH):
    shift = ((n - n % 2) // 2) * (d-a)
    atoms += f"H {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mol.build()

mf = scf.RHF(mol)#.density_fit()
e = mf.kernel()

mycc = cc.CCSD(mf)
e = mycc.kernel()
#mycc.t1 = mycc.t1*0

options = {'n_eql': 4,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 10,
            'n_blocks': 10,
            'n_walkers': 40,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc import pyscf_interface
from ad_afqmc.cisd_perturb import sample_pt2
pyscf_interface.prep_afqmc(mycc,chol_cut=1e-7)

sample_pt2.run_afqmc_ccsd_pt(options,nproc=10)
