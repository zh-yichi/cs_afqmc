import sys
from pyscf.lib import logger
from ad_afqmc.lno_ccsd import lno_ccsd
from pyscf import gto, scf, mp
from ad_afqmc.lno.afqmc import LNOAFQMC

log = logger.Logger(sys.stdout, 6)

atoms = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161
'''

mol = gto.M(atom=atoms, basis="ccpvdz", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

frozen = 1

options = {'n_eql': 5,
           'n_prop_steps': 30,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 20,
            'n_walkers': 50,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.01,
            'ad_mode':None,
            }


# thresh = 1.00e-04
# lno_ccsd.run_lno_ccsd_afqmc(mf,thresh,frozen,options,1e-6,8)

mmp = mp.MP2(mf, frozen=frozen)
mmp.kernel()

filename = 'fragmentenergies.txt'
# LNO
for thresh in [1e-4]:
    f = open(filename,'w')
    f.close()
    mfcc = LNOAFQMC(mf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = 1e-3
    mfcc.thresh_vir = thresh
    mfcc.nblocks = 20
    mfcc.seed = 1234 
    mfcc.lo_type = 'pm'
    mfcc.no_type = 'cim'
    mfcc.frag_lolist = '1o'
    mfcc.nwalk_per_proc = 30
    mfcc.force_outcore_ao2mo = True
    mfcc.kernel()#canonicalize=False,chol_vecs=chol_vecs)
    ecc = mfcc.e_corr #+ mf.energy_nuc()
    ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)

    log.info('thresh = %.0e  E_corr(AFQMC)     = %.10f, %.10f  rel(wrt CCSD) = %6.2f%%  '
             'diff = % .10f',
             thresh, ecc,mf.energy_nuc(), 0, 0)
    log.info('E_corr(AFQMC+PT2) = %.10f',ecc_pt2corrected)