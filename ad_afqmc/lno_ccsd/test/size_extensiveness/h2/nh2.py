import numpy as np
from pyscf import gto, scf, cc

nh2 = 1
s = 20
e_hf = np.zeros(nh2)
e_ccsd = np.zeros(nh2)

def h2(d):
    h2 = f'''
    H 0 {d} 0
    H 1.058354498 {d} 0
    '''
    return h2

atoms = ''
for i in range(nh2):
    atoms += h2(i*s)

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()
e_hf[i] = mf.e_tot

mycc = cc.CCSD(mf)
mycc.kernel()
e_ccsd[i] = mycc.e_tot

from ad_afqmc.lno_ccsd import lno_ccsd

options = {'n_eql': 6,
           'n_prop_steps': 30,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 20,
            'n_walkers': 50,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'ad_mode':None,
            'use_gpu': False,
            }


thresh = 1.00e-05
lno_ccsd.run_lno_ccsd_afqmc(mf,thresh,[],options,1e-6,8,mp2=True)
