import sys
import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib

from ad_afqmc.lno.base import LNO
from ad_afqmc.lno.cc import LNOCCSD
_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
einsum = lib.einsum

from pyscf import gto, scf, mp, cc

log = logger.Logger(sys.stdout, 6)

a = 1.831
nH = 40
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"
basis = 'sto6g'

mol = gto.M(atom=atoms, basis=basis, unit='Bohr', symmetry=False)
mol.verbose = 4

mf = scf.RHF(mol).density_fit()
mf.kernel()

from pyscf import lo
mo = mf.mo_coeff
nocc = np.count_nonzero(mf.mo_occ)
occloc = lo.Boys(mol,mo[:,:nocc]).kernel()
virloc = lo.Boys(mol,mf.mo_coeff[:,nocc:]).kernel()
mo_loc = np.hstack((occloc,virloc))
mf.mo_coeff = mo_loc

frozen = 0
# canonical
# mmp = mp.MP2(mf, frozen=frozen)
# mmp.kernel()

mcc = cc.CCSD(mf, frozen=frozen).set(verbose=5)
eris = mcc.ao2mo()
mcc.kernel(eris=eris)

# from pyscf.cc.ccsd_t import kernel as CCSD_T
# eccsd_t = CCSD_T(mcc, eris)

# LNO
for thresh in [1e-4,1e-5,1e-6,1e-7]:
    mfcc = LNOCCSD(mf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = thresh*10
    mfcc.thresh_vir = thresh
    mfcc.lo_type = 'boys'
    mfcc.no_type = 'cim'
    mfcc.frag_lolist = '1o'
    # mfcc.ccsd_t = True
    mfcc.force_outcore_ao2mo = True
    mfcc.kernel(canonicalize=True)
    ecc = mfcc.e_corr
    # ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
    ecc_t = mfcc.e_corr_ccsd_t
    log.info('thresh = %.0e  E_corr(CCSD)     = %.10f  rel = %6.2f%%  '
             'diff = % .10f',
             thresh, ecc, ecc/mcc.e_corr*100, ecc-mcc.e_corr)
    # log.info('                E_corr(CCSD+PT2) = %.10f  rel = %6.2f%%  '
    #          'diff = % .10f',
    #          ecc_pt2corrected, ecc_pt2corrected/mcc.e_corr*100,
    #          ecc_pt2corrected-mcc.e_corr)
    # log.info('                E_corr(CCSD_T)   = %.10f  rel = %6.2f%%  '
    #          'diff = % .10f',
    #          ecc_t, ecc_t/eccsd_t*100, ecc_t-eccsd_t)
