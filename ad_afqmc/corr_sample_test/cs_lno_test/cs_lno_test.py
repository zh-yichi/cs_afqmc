import sys
# import os
# import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib

# from ad_afqmc.lno.base import LNO
from ad_afqmc.lno.afqmc import LNOAFQMC
# from ad_afqmc import config
#from lno.afqmc import mpi_jax

from pyscf import gto, scf, mp, cc

log = logger.Logger(sys.stdout, 6)

atom = '''
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
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

# mcc = cc.CCSD(mf, frozen=frozen).set(verbose=5)
# eris = mcc.ao2mo()
# mcc.kernel(eris=eris)
# from pyscf.cc.ccsd_t import kernel as CCSD_T
# eccsd_t = CCSD_T(mcc, eris)

filename = 'fragmentenergies.txt'

# LNO
for thresh in [1e-4]:
    f = open(filename,'w')
    f.close()
    mfcc = LNOAFQMC(mf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = thresh*10
    mfcc.thresh_vir = thresh
    mfcc.nblocks = 10
    mfcc.nproc = 5
    mfcc.nwalk_per_proc = 20
    mfcc.lo_type = 'pm'
    mfcc.no_type = 'cim'
    mfcc.frag_lolist = '1o'
    mfcc.chol_cut = 1e-5
    #mfcc.force_outcore_ao2mo = True
    #mfcc.runfrags = [17,18,19,20,21,22,23,24,25,26,27,28,29]
    #print("using gpu: ",config.afqmc_config["use_gpu"])
    mfcc.kernel()
    ecc = mfcc.e_corr
    ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
    log.info('thresh = %.0e  E_corr(AFQMC) = %.10f  E_corr(AFQMC_P2) = %.10f', thresh, ecc, ecc_pt2corrected)