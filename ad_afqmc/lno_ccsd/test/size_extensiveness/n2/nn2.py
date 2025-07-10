import numpy as np
from pyscf import gto, scf, cc

nn2 = 1
s = 20
e_hf = np.zeros(nn2)
e_ccsd_t = np.zeros(nn2)

def n2(d):
    n2 = f'''
    N 0 {d} 0
    N 1.12027 {d} 0
    '''
    return n2

atoms = ''
for i in range(nn2):
    atoms += n2(i*s)

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()
e_hf[i] = mf.e_tot

# mycc = cc.CCSD(mf)
# mycc.kernel()
# e_ccsd_t[i] = mycc.e_tot

from ad_afqmc.lno_ccsd import lno_ccsd

options = {'n_eql': 3,
           'n_prop_steps': 50,
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
lno_ccsd.run_lno_ccsd_afqmc(mf,thresh,2,options,1e-5,8,mp2=True)