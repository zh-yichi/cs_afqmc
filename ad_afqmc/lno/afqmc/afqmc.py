import sys
import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib
#from pyscf.shciscf import shci
#import QMCUtils
from lno.base import LNO
from lno.cc import ccsd

_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
einsum = lib.einsum
THRESH_INTERNAL = 1e-10
def impurity_solve(mf, mo_coeff, lo_coeff, ccsd_t=False, eris=None, frozen=None,
                   log=None, verbose_imp=0,filename='fragmentenergies.txt',chol_vecs=None,can_orbfrag=None,nblocks = 800,seed=None,chol_cut = None,cholesky_threshold = 2e-3,vmc_root = None,maxError=2e-4,dt=0.005,multislater=False,DiceBinary=None,ndets=1000,nwalk_per_proc=20,active_space=[],nproc=None):
    r''' Solve impurity problem and calculate local correlation energy.

    Args:
        mo_coeff (np.ndarray):
            MOs where the impurity problem is solved.
        lo_coeff (np.ndarray):
            LOs which the local correlation energy is calculated for.
        ccsd_t (bool):
            If True, CCSD(T) energy is calculated and returned as the third
            item (0 is returned otherwise).
        frozen (int or list; optional):
            Same syntax as `frozen` in MP2, CCSD, etc.

    Return:
        e_loc_corr_pt2, e_loc_corr_ccsd, e_loc_corr_ccsd_t:
            Local correlation energy at MP2, CCSD, and CCSD(T) level. Note that
            the CCSD(T) energy is 0 unless 'ccsd_t' is set to True.
    '''
    print("Solving impurity problem")
    log = logger.new_logger(mf if log is None else log)
    cput1 = (logger.process_clock(), logger.perf_counter())

    maskocc = mf.mo_occ>1e-10
    nocc = np.count_nonzero(maskocc)
    nmo = mf.mo_occ.size

    # Convert frozen to 0 bc PySCF solvers do not support frozen=None or empty list
    if frozen is None:
        frozen = 0
    elif isinstance(frozen, (list,tuple,np.ndarray)) and len(frozen) == 0:
        frozen = 0

    if isinstance(frozen, (int,np.integer)):
        maskact = np.hstack([np.zeros(frozen,dtype=bool),
                             np.ones(nmo-frozen,dtype=bool)])
    elif isinstance(frozen, (list,tuple,np.ndarray)):
        maskact = np.array([i not in frozen for i in range(nmo)])
    else:
        raise RuntimeError
    #import pdb
    #pdb.set_trace()
    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]

    # solve impurity problem
    s1e = mf.get_ovlp() if eris is None else eris.s1e
    prjlo = fdot(lo_coeff.T, s1e, orbactocc)

    #print("DQMC root: ",vmc_root)


    orbitalE= nactocc-1
    act_index = [] 
    norb_act = (nactocc+nactvir)
    nelec_act = nactocc*2
    central_mo = mo_coeff[:,orbitalE].copy()
    for i in range(len(active_space)):
      act_index.append(np.argmax(np.abs(mf.mo_coeff[:,active_space[i]]@mf.get_ovlp(mf.mol)@mo_coeff)))

    




   
#    #import QMCUtils
#    import LNOhelper as QMCUtils
    from ad_afqmc import run_afqmc
    e_afqmc,err_afqmc = run_afqmc.run_afqmc_lno_mf(mf,mo_coeff = mo_coeff,norb_act = (nactocc+nactvir),nelec_act = nactocc*2,norb_frozen=frozen,nwalk_per_proc = nwalk_per_proc,orbitalE=nactocc-1,vmc_root=vmc_root,nblocks=nblocks,chol_vecs=chol_vecs,seed=seed,chol_cut = chol_cut, cholesky_threshold = cholesky_threshold,maxError=maxError,dt=dt,nproc=nproc,prjlo=prjlo)#,right='uhf')
#    e_afqmc,err_afqmc = QMCUtils.run_afqmc_lno_mc(mf,mo_coeff = mo_sort,norb_act = norb_act,nelec_act = nelec_act,norb_frozen=frz_ind,nwalk_per_proc = 20,orbitalE=orbitalE,vmc_root=vmc_root,nblocks=nblocks,chol_vecs=chol_vecs,seed=seed,chol_cut = chol_cut, cholesky_threshold = cholesky_threshold,maxError=maxError,dt=dt,prjlo=prjlo,active_space=act_index,ndets=ndets)#,ncore=ncore)

    #e_afqmc,err_afqmc = QMCUtils.run_afqmc_lno_mf(mf,mo_coeff = mo_coeff,norb_act = (nactocc+nactvir),nelec_act = nactocc*2,norb_frozen=frozen,nwalk_per_proc = 20,orbitalE=nactocc-1,vmc_root=vmc_root,nblocks=nblocks,chol_vecs=chol_vecs,seed=seed,chol_cut = chol_cut, cholesky_threshold = cholesky_threshold,maxError=maxError,dt=dt,prjlo=prjlo)#,right='uhf')
    #e_afqmc,err_afqmc = QMCUtils.run_afqmc_lno_mf(mf,mo_coeff = can_orbfrag,norb_act = (nactocc+nactvir),nelec_act = nactocc*2,norb_frozen=frozen,nwalk_per_proc = 20,orbitalE=nactocc-1,vmc_root=vmc_root,nblocks=nblocks,chol_vecs=chol_vecs,seed=seed,chol_cut = chol_cut, cholesky_threshold = cholesky_threshold,maxError=maxError,dt=dt,prjlo=prjlo)#,right='uhf')


    

    if(e_afqmc==0 and err_afqmc==0):
        frag_msg = '  '.join([f'E_corr(MP2) = 0', f'E_corr(AFQMC) = 0', f'E_corr(CCSD(T)) = 0'])
        return frag_msg, (0, 0, 0)
 

    #MP2 correction
    from pyscf.cc import CCSD
    s1e = mf.get_ovlp() if eris is None else eris.s1e
    prjlo = fdot(lo_coeff.T, s1e, can_orbfrag[:, maskact& maskocc])
    mcc = CCSD(mf, mo_coeff=can_orbfrag, frozen=frozen).set(verbose=verbose_imp)
    mcc.ao2mo = ccsd.ccsd_ao2mo.__get__(mcc, mcc.__class__)
    mcc._s1e = s1e
    if eris is not None:
        mcc._h1e = eris.h1e
        mcc._vhf = eris.vhf
    imp_eris = mcc.ao2mo()
    #import pdb
    #pdb.set_trace()
    if isinstance(imp_eris.ovov, np.ndarray):
        ovov = imp_eris.ovov
    else:
        ovov = imp_eris.ovov[()]
    oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
    ovov = None
    cput1 = log.timer_debug1('imp sol - eri    ', *cput1)
    # MP2 fragment energy
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    cput1 = log.timer_debug1('imp sol - mp2 amp', *cput1)
    elcorr_pt2 = get_fragment_energy(oovv, t2, prjlo)
    #print(elcorr_pt2)
    #import pdb
    #pdb.set_trace()

    f = open(filename,'a')
    f.write(f'{e_afqmc}     \t {err_afqmc}     \t {elcorr_pt2}\n')
    f.close()
  
    frag_msg = '  '.join([f'E_corr(MP2) = {elcorr_pt2}', f'E_corr(AFQMC) = {e_afqmc:.15g}', f'E_corr(CCSD(T)) = 0'])

    return frag_msg, (elcorr_pt2, e_afqmc, 0)

def get_fragment_energy(oovv, t2, prj):
    m = fdot(prj.T, prj)
    return einsum('ijab,kjab,ik->',t2,2*oovv-oovv.transpose(0,1,3,2),m)


class LNOAFQMC(LNO):

    def __init__(self, mf, thresh=1e-6, frozen=None):

        LNO.__init__(self, mf, thresh=thresh, frozen=frozen)

        self.efrag_cc = None
        self.efrag_pt2 = None
        self.efrag_cc_t = None
        self.ccsd_t = False
        self.nblocks = 800
        self.seed = None
        self.chol_cut = 1e-6
        self.cholesky_threshold = 0.5e-3
        self.chol_vecs = None
        self.vmc_root = None #'/projects/joku8258/software/VMC_forwork/VMC/'
        self.multislater=False
        self.DiceBinary= None #"/projects/joku8258/software/alpine_software/Dice_Dice/Dice/bin/Dice" #'/projects/xuwa0145/tools/Dice/bin_gcc_alpine/Dice'
        self.nwalk_per_proc = 20
        self.ndets=1000
        self.maxError = 1e-3
        self.dt = 0.005
        self.runfrags = []
        self.active_space = []
        self.active_space_mo = []
        self.nproc=None

    def dump_flags(self, verbose=None):
        LNO.dump_flags(self, verbose=verbose)
        return self

    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris=None, frozen=None, log=None,can_orbfrag=None):
        return impurity_solve(mf, mo_coeff, lo_coeff, eris=eris, frozen=frozen, log=log,verbose_imp=self.verbose_imp, ccsd_t=self.ccsd_t,chol_vecs=self.chol_vecs,can_orbfrag=can_orbfrag,nblocks=self.nblocks,seed = self.seed,chol_cut = self.chol_cut,cholesky_threshold = self.cholesky_threshold,vmc_root = self.vmc_root,maxError= self.maxError,dt=self.dt,multislater=self.multislater,DiceBinary=self.DiceBinary,ndets=self.ndets,nwalk_per_proc=self.nwalk_per_proc,active_space=self.active_space,nproc=self.nproc)

    def _post_proc(self, frag_res, frag_wghtlist):
        ''' Post processing results returned by `impurity_solve` collected in `frag_res`.
        '''
        nfrag = len(frag_res)
        efrag_pt2 = np.zeros(nfrag)
        efrag_cc = np.zeros(nfrag)
        efrag_cc_t = np.zeros(nfrag)
        for i in range(nfrag):
            efrag_pt2[i], efrag_cc[i], efrag_cc_t[i] = frag_res[i]
        #print(sum(efrag_pt2))
        #import pdb
        #pdb.set_trace()
        self.efrag_pt2  = efrag_pt2  * frag_wghtlist
        self.efrag_cc   = efrag_cc   * frag_wghtlist
        self.efrag_cc_t = efrag_cc_t * frag_wghtlist

    def _finalize(self):
        r''' Hook for dumping results and clearing up the object.'''
        #logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
        #            'LNOMP2', self.e_tot_pt2, self.e_corr_pt2)
        logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
                    'LNO-AFQMC', self.e_tot, self.e_corr)
        if self.ccsd_t:
            logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
                        'LNOCCSD_T', self.e_tot_ccsd_t, self.e_corr_ccsd_t)
        return self

    @property
    def e_corr(self):
        return self.e_corr_ccsd

    @property
    def e_corr_ccsd(self):
        e_corr = np.sum(self.efrag_cc)
        return e_corr

    @property
    def e_corr_pt2(self):
        e_corr = np.sum(self.efrag_pt2)
        return e_corr

    @property
    def e_corr_ccsd_t(self):
        e_corr = np.sum(self.efrag_cc_t)
        return e_corr

    @property
    def e_tot_ccsd(self):
        return self.e_corr_ccsd + self._scf.e_tot

    @property
    def e_tot_pt2(self):
        return self.e_corr_pt2 + self._scf.e_tot

    def e_corr_pt2corrected(self, ept2):
       # print(self.e_corr,self.e_corr_pt2,ept2)
       # import pdb
       # pdb.set_trace()
        return self.e_corr - self.e_corr_pt2 + ept2

    def e_tot_pt2corrected(self, ept2):
        return self._scf.e_tot + self.e_corr_pt2corrected(ept2)

    def e_corr_ccsd_pt2corrected(self, ept2):
        return self.e_corr_ccsd - self.e_corr_pt2 + ept2

    def e_tot_ccsd_pt2corrected(self, ept2):
        return self._scf.e_tot_ccsd + self.e_corr_pt2corrected(ept2)

    def e_corr_ccsd_t_pt2corrected(self, ept2):
        return self.e_corr_ccsd_t - self.e_corr_pt2 + ept2

    def e_tot_ccsd_t_pt2corrected(self, ept2):
        return self._scf.e_tot_ccsd_t + self.e_corr_pt2corrected(ept2)




if __name__ == '__main__':
    from pyscf import gto, scf, mp, cc

    log = logger.Logger(sys.stdout, 6)

    # S22-2: water dimer
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
    

# canonical
    mmp = mp.MP2(mf, frozen=frozen)
    mmp.kernel()
   # import pdb
   # pdb.set_trace()

    mcc = cc.CCSD(mf, frozen=frozen).set(verbose=5)
    eris = mcc.ao2mo()
    mcc.kernel(eris=eris)
    from pyscf.cc.ccsd_t import kernel as CCSD_T
    eccsd_t = CCSD_T(mcc, eris)
    filename = 'fragmentenergies.txt'
    chol_cut = 1e-6
    cholesky_threshold = 0
    chol_vecs =None# QMCUtils.chunked_cholesky(mol, max_error=chol_cut, verbose=True)
    #vmc_root = '/projects/joku8258/software/DICE_LNO/Dice' #'/projects/joku8258/software/VMC_forwork/VMC/'
    #vmc_root = '/projects/joku8258/software/VMC_forwork/VMC/'
    #vmc_root = '/projects/joku8258/software/alpine_software/Dice'
# LNO
    for thresh in [1e-4,1e-5]:
        f = open(filename,'w')
        f.close()
        mfcc = LNOAFQMC(mf, thresh=thresh, frozen=frozen).set(verbose=5)
        mfcc.thresh_occ = thresh*10
        mfcc.thresh_vir = thresh
        mfcc.nblocks = 10
        #mfcc.seed = 1234
        mfcc.chol_cut = chol_cut
        mfcc.cholesky_threshold = cholesky_threshold
        mfcc.lo_type = 'pm'
        #mfcc.runfrags=[2,7]
        #mfcc.maxError=3e-3
        mfcc.dt = 0.001
        #mfcc.multislater=True
        mfcc.no_type = 'cim'
        mfcc.frag_lolist = '1o'
        mfcc.force_outcore_ao2mo = True
        mfcc.chol_vecs = chol_vecs
        #mfcc.vmc_root = vmc_root
        mfcc.kernel()
        ecc = mfcc.e_corr #+ mf.energy_nuc()
        ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
        ecc_t = mfcc.e_corr_ccsd_t
        import os
        os.system(f'mv {filename} {filename}_{thresh};')
        log.info('thresh = %.0e  E_corr(AFQMC)     = %.10f, %.10f  rel(wrt CCSD) = %6.2f%%  '
                 'diff = % .10f',
                 thresh, ecc,mf.energy_nuc(), ecc/mcc.e_corr*100, ecc-mcc.e_corr)
        log.info('                E_corr(AFQMC+PT2) = %.10f  rel = %6.2f%%  '
                 'diff = % .10f',
                 ecc_pt2corrected, ecc_pt2corrected/mcc.e_corr*100,
                 ecc_pt2corrected-mcc.e_corr)






