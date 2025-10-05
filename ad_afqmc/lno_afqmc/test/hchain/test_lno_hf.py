from ad_afqmc.lno_afqmc import norhf_test
from ad_afqmc.lno.afqmc import LNOAFQMC
from pyscf import gto, scf, cc, mp
import numpy as np
import os, sys
from pyscf.lib import logger
log = logger.Logger(sys.stdout, 6)

a = 1.
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

# mycc = cc.CCSD(mf)
# mycc.kernel()

# def thouless_trans(t1):
#     q, r = np.linalg.qr(t1)
#     u_ai = r.T
#     u_ji = q
#     u_occ = np.vstack((u_ji,u_ai))
#     u, _, _ = np.linalg.svd(u_occ)
#     return u
# u = thouless_trans(10*mycc.t1)
# mo_t = mf.mo_coeff @ u
# mf.mo_coeff = mo_t

# myci = ci.CISD(mf)
# myci.kernel()

# options = {'n_eql': 4,
#            'n_prop_steps': 50,
#             'n_ene_blocks': 1,
#             'n_sr_blocks': 1,
#             'n_blocks': 10,
#             'n_walkers': 5,
#             'seed': 2,
#             'walker_type': 'rhf',
#             'trial': 'rhf',
#             'dt':0.005,
#             'ad_mode':None,
#             'use_gpu': False,
#             'which_rhf': 1,
#             }

# threshs = [1e-4]
# for i,thresh in enumerate(threshs):
#     norhf_test.run_lno_afqmc_norhf_test(mf,thresh,[],options,nproc=5)
#     os.system(f"mv results.out results.out1")

# norhf_test.sum_results_norhf_test(1)

frozen = 0
mmp = mp.MP2(mf, frozen=frozen)
mmp.kernel()

filename = 'fragmentenergies.txt'

for thresh in [1e-4]:
    f = open(filename,'w')
    f.close()
    mfcc = LNOAFQMC(mf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = 10*thresh
    mfcc.thresh_vir = thresh
    mfcc.nblocks = 100
    mfcc.seed = 1234
    mfcc.lo_type = 'pm'
    mfcc.no_type = 'cim'
    mfcc.frag_lolist = '1o'
    mfcc.nwalk_per_proc = 30
    mfcc.nproc = 8
    mfcc.force_outcore_ao2mo = True
    mfcc.kernel()#canonicalize=False,chol_vecs=chol_vecs)
    ecc = mfcc.e_corr
    ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)

    log.info('thresh = %.2e  E_corr(AFQMC) = %.6f, E(AFQMC) = %.6f ',
             thresh, ecc, ecc+mf.e_tot)
    log.info('E_corr(AFQMC+PT2) = %.10f',ecc_pt2corrected)
