import sys
import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib
#import QMCUtils
from lno.base_test import LNO
#from lno.afqmc import LNOAFQMC
from lno.cc import LNOCCSD

from pyscf import gto, scf, mp, cc

log = logger.Logger(sys.stdout, 6)



atom = '''
C      1.232702   -0.335203    1.255044
C     -0.157003    0.158960    1.705377
C     -1.210951   -0.238600    0.675301
C     -0.795133    0.254804   -0.710134
C      0.619545   -0.251960   -1.050584
O      1.537869    0.195518   -0.033765
O     -0.542301   -0.413660    2.957259
O     -2.497664    0.324315    0.962614
O     -1.703190   -0.207685   -1.719754
C      1.137062    0.287402   -2.392608
O      0.516091   -0.339309   -3.509595
H      0.184686   -0.255139    3.577646
H     -2.750235    0.014238    1.843888
H     -0.442052   -0.264335   -3.373345
H     -2.592981    0.046961   -1.434405
O      2.236387    0.057464    2.156426
H      2.318678    1.022257    2.086875
H      1.251535   -1.436912    1.239345
H     -0.117417    1.261339    1.770300
H     -1.274776   -1.339658    0.650191
H     -0.769446    1.357354   -0.697205
H      0.608308   -1.354399   -1.090327
H      1.001164    1.383982   -2.407621
H      2.209122    0.072265   -2.460921
'''

basis = 'cc-pvdz'

mol = gto.M(atom=atom, basis=basis)
mol.verbose = 4
mol.symmetry = True
mol.spin = 0 # This is the difference in \alpha and \beta electrons so a value of 2 indicates a triplet.
mol.charge = 0
mol.verbose = 4
mol.max_memory = 110000
mol.build()


mf = scf.RHF(mol).density_fit()
mf.max_cycle = 200
mf.kernel()

frozen = 12
    
#QMCUtils.run_afqmc(mf,norb_frozen = 10,nwalk_per_proc=15,nblocks = 2500)
#exit(0)
## canonical
mmp = mp.MP2(mf, frozen=frozen)
mmp.kernel()
#
#mcc = cc.CCSD(mf, frozen=frozen).set(verbose=5)
#eris = mcc.ao2mo()
#mcc.kernel(eris=eris)
#from pyscf.cc.ccsd_t import kernel as CCSD_T
#eccsd_t = CCSD_T(mcc, eris)
filename = 'fragmentenergies.txt'
chol_vecs = None#QMCUtils.chunked_cholesky(mol, max_error=1e-5, verbose=False)

chol_cut = 1e-4
cholesky_threshold = 0
# LNO
#frags = [i for i in range(mmp.nocc)]
#n = 2
#runfrags= [frags[i*int(len(frags)/n):(i+1)*int(len(frags)/n)] if i<n-1 else frags[i*int(len(frags)/n):] for i in range(n)]

for thresh in [1e-4]:
    f = open(filename,'w')
    f.close()
    mfcc = LNOCCSD(mf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = thresh*10 #1e-3
    mfcc.thresh_vir = thresh
#    mfcc.chol_cut = chol_cut
#    mfcc.cholesky_threshold = cholesky_threshold
    mfcc.lo_type = 'pm'
    mfcc.no_type = 'cim'
#    mfcc.nblocks = 1500
#    mfcc.maxError = 1e-3
    mfcc.frag_lolist = '1o'
#    mfcc.runfrags = runfrags
    mfcc.force_outcore_ao2mo = True
    mfcc.kernel()
    ecc = mfcc.e_corr #+ mf.energy_nuc()
    ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
#    import os
#    os.system(f'mv {filename} {filename}_v_{mfcc.thresh_vir}_o_{mfcc.thresh_occ}')
    #ecc_t = mfcc.e_corr_ccsd_t
    log.info('thresh = %.0e  E_corr(AFQMC)     = %.10f rel(wrt CCSD) = %6.2f%%  '
             'diff = % .10f',
             thresh, ecc, 0, 0)
    #log.info('                E_corr(AFQMC+PT2) = %.10f  rel = %6.2f%%  '
    #             'diff = % .10f',
    #             ecc_pt2corrected, ecc_pt2corrected/mcc.e_corr*100,
    #             ecc_pt2corrected-mcc.e_corr)
    log.info('                E_corr(AFQMC+PT2) = %.10f  rel = %6.2f%%  '
                 'diff = % .10f',
                 ecc_pt2corrected, 0,
                 0)




