import sys
from pyscf.lib import logger

from ad_afqmc.lno.afqmc import LNOAFQMC

from pyscf import gto, scf, mp

log = logger.Logger(sys.stdout, 6)

atom = '''
 O   -1.485163346097   -0.114724564047    0.000000000000
 H   -1.868415346097    0.762298435953    0.000000000000
 H   -0.533833346097    0.040507435953    0.000000000000
 O    1.416468653903    0.111264435953    0.000000000000
 H    1.746241653903   -0.373945564047   -0.758561000000
 H    1.746241653903   -0.373945564047    0.758561000000
'''
basis = 'sto6g'
mol = gto.M(atom=atom, basis=basis)
mol.verbose = 4
mf = scf.RHF(mol).density_fit()
mf.kernel()

frozen = 2
## canonical
mmp = mp.MP2(mf, frozen=frozen)
mmp.kernel()
#
# mcc = cc.CCSD(mf, frozen=frozen).set(verbose=5)
# eris = mcc.ao2mo()
# mcc.kernel(eris=eris)
# from pyscf.cc.ccsd_t import kernel as CCSD_T
# eccsd_t = CCSD_T(mcc, eris)

filename = './fragmentenergies.txt'
for thresh in [1e-4]:
    f = open(filename,'w')
    f.close()
    mfcc = LNOAFQMC(mf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = thresh*10
    mfcc.thresh_vir = thresh
    mfcc.nblocks = 10
    mfcc.nwalk_per_proc = 10
    mfcc.nproc = 4
    mfcc.lo_type = 'pm'
    mfcc.no_type = 'cim'
    mfcc.frag_lolist = '1o'
    mfcc.chol_cut = 1e-6
    #mfcc.force_outcore_ao2mo = True
    #mfcc.runfrags = [17,18,19,20,21,22,23,24,25,26,27,28,29]
    mfcc.kernel()
    ecc = mfcc.e_corr
    ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)

    log.info('thresh = %.0e  E_corr(AFQMC) = %.10f  E_corr(AFQMC_P2) = %.10f', thresh, ecc, ecc_pt2corrected)





