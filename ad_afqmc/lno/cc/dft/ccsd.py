import sys
import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib, dft

from lno.base import LNO
from lno.cc import LNOCCSD
_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
einsum = lib.einsum





from pyscf import gto, scf, mp, cc

log = logger.Logger(sys.stdout, 6)

# S22-2: water dimer
atom = '''
 O   -1.485163346097   -0.114724564047    0.000000000000
 H   -1.868415346097    0.762298435953    0.000000000000
 H   -0.533833346097    0.040507435953    0.000000000000
 O    1.416468653903    0.111264435953    0.000000000000
 H    1.746241653903   -0.373945564047   -0.758561000000
 H    1.746241653903   -0.373945564047    0.758561000000
'''
basis = 'cc-pvdz'

mol = gto.M(atom=atom, basis=basis)
mol.verbose = 4

mf = scf.RHF(mol).density_fit()
mf.kernel()

frozen = 2

mf_dft = dft.RKS(mol).density_fit()
mf_dft.xc = 'b3lyp'
mf_dft.kernel()

mf.mo_coeff = 1.0*mf_dft.mo_coeff
# canonical

#mf_hf = mol.HF()
#if getattr(mf_dft, "with_x2c", False):
#    mf_hf = mf_hf.x2c()
#mf_hf.__dict__.update(mf_dft.__dict__)
#
##mf = mf_hf
mf_hf = mf
mmp = mp.MP2(mf_hf, frozen=frozen)
mmp.kernel()

mcc = cc.CCSD(mf_hf, frozen=frozen).set(verbose=5)
eris = mcc.ao2mo()
mcc.kernel(eris=eris)

from pyscf.cc.ccsd_t import kernel as CCSD_T
eccsd_t = CCSD_T(mcc, eris)
#mf.mo_coeff = mf_dft.mo_coeff
# LNO
for thresh in [1e-3,1e-4,1e-5]:
    mfcc = LNOCCSD(mf_hf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = 1e-3
    mfcc.thresh_vir = thresh
    mfcc.lo_type = 'pm'
    mfcc.no_type = 'cim'
    mfcc.frag_lolist = '1o'
    mfcc.ccsd_t = True
    mfcc.force_outcore_ao2mo = True
    mfcc.kernel(canonicalize=True)
    ecc = mfcc.e_corr
    ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
    ecc_t = mfcc.e_corr_ccsd_t
    ecc_ccsdt_pt2corrected = mfcc.e_corr_ccsd_t_pt2corrected(mmp.e_corr)
    print("CCSD(T)+PT2=",ecc_ccsdt_pt2corrected)
    log.info('thresh = %.0e  E_corr(CCSD)     = %.10f  rel = %6.2f%%  '
             'diff = % .10f',
             thresh, ecc, ecc/mcc.e_corr*100, ecc-mcc.e_corr)
    log.info('                E_corr(CCSD+PT2) = %.10f  rel = %6.2f%%  '
             'diff = % .10f',
             ecc_pt2corrected, ecc_pt2corrected/mcc.e_corr*100,
             ecc_pt2corrected-mcc.e_corr)
    log.info('                E_corr(CCSD_T)   = %.10f  rel = %6.2f%%  '
             'diff = % .10f',
             ecc_t, ecc_t/eccsd_t*100, ecc_t-eccsd_t)
