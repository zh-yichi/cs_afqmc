import sys
import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib
import QMCUtils
from lno.base import LNO
from lno.afqmc import LNOAFQMC

from pyscf import gto, scf, mp, cc

log = logger.Logger(sys.stdout, 6)

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
    

## canonical
mmp = mp.MP2(mf, frozen=frozen)
mmp.kernel()
#
mcc = cc.CCSD(mf, frozen=frozen).set(verbose=5)
eris = mcc.ao2mo()
mcc.kernel(eris=eris)
from pyscf.cc.ccsd_t import kernel as CCSD_T
eccsd_t = CCSD_T(mcc, eris)
filename = 'fragmentenergies.txt'
chol_vecs = QMCUtils.chunked_cholesky(mol, max_error=1e-5, verbose=False)
vmc_root = '/projects/joku8258/software/alpine_software/Dice'
# LNO
for thresh in [1e-4]:
    f = open(filename,'w')
    f.close()
    mfcc = LNOAFQMC(mf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = 1e-3
    mfcc.thresh_vir = thresh
    mfcc.nblocks = 250
    mfcc.nwalk_per_proc = 10
#    mfcc.seed = 1234 
#    mfcc.runfrags=[2]
    mfcc.lo_type = 'pm'
    mfcc.no_type = 'cim'
    mfcc.frag_lolist = '1o'
    mfcc.force_outcore_ao2mo = True
#    mfcc.vmc_root = vmc_root
    mfcc.kernel()#canonicalize=False,chol_vecs=chol_vecs)
    ecc = mfcc.e_corr #+ mf.energy_nuc()
    ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
    #ecc_t = mfcc.e_corr_ccsd_t
#    log.info('thresh = %.0e  E_corr(AFQMC)     = %.10f, %.10f  rel(wrt CCSD) = %6.2f%%  '
#             'diff = % .10f',
#             thresh, ecc,mf.energy_nuc(), ecc/mcc.e_corr*100, ecc-mcc.e_corr)
    log.info('thresh = %.0e  E_corr(AFQMC)     = %.10f, %.10f  rel(wrt CCSD) = %6.2f%%  '
             'diff = % .10f',
             thresh, ecc,mf.energy_nuc(), 0, 0)
    log.info('                E_corr(AFQMC+PT2) = %.10f  rel = %6.2f%%  '
                 'diff = % .10f',
                 ecc_pt2corrected, ecc_pt2corrected/mcc.e_corr*100,
                 ecc_pt2corrected-mcc.e_corr)





