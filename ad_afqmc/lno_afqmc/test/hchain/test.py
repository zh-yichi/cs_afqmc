from ad_afqmc.lno_afqmc import norhf_test
from pyscf import gto, scf, cc
import numpy as np
import os

a = 1.
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

def thouless_trans(t1):
    q, r = np.linalg.qr(t1)
    u_ai = r.T
    u_ji = q
    u_occ = np.vstack((u_ji,u_ai))
    u, _, _ = np.linalg.svd(u_occ)
    return u
u = thouless_trans(10*mycc.t1)
mo_t = mf.mo_coeff @ u
mf.mo_coeff = mo_t

# myci = ci.CISD(mf)
# myci.kernel()

options = {'n_eql': 4,
           'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 1,
            'n_blocks': 10,
            'n_walkers': 5,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'rhf',
            'dt':0.005,
            'ad_mode':None,
            'use_gpu': False,
            'which_rhf': 1,
            }

threshs = [1e-4]
for i,thresh in enumerate(threshs):
    norhf_test.run_lno_afqmc_norhf_test(mf,thresh,[],options,nproc=5)
    os.system(f"mv results.out results.out1")

norhf_test.sum_results_norhf_test(1)
