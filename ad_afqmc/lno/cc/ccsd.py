import sys
import numpy as np
from functools import reduce
from ad_afqmc.lno.tools import tools
from pyscf.lib import logger
from pyscf import lib
#from lno.base_test import LNO
from ad_afqmc.lno.base import LNO

_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
einsum = lib.einsum


r''' TODO's
[x] customized CCSD to make PBC work
[x] (T)
'''

''' Beginning of modification of PySCF's CCSD class

    The following functions are modified from pyscf.cc module

    In PySCF, 1e integrals (s1e, h1e, vhf) are calculated whenever a CCSD object is
    initialized. In LNOCCSD, this means that the same set of 1e integrals are evaluated
    for every fragment. For PBC calculations, evaluating 1e integrals (especially h1e
    and vhf) can be very slow in PySCF's current implementation.

    The following modification forces the CCSD class to take precomputed 1e integrals
    and thus can lead to significant amount of time saving in PBC LNOCCSD calculations.
'''
from pyscf import cc
class _ChemistsERIs(cc.ccsd._ChemistsERIs):
    def _common_init_(self, mycc, mo_coeff=None):
        from pyscf.mp.mp2 import _mo_without_core

        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

# Note: Recomputed fock matrix and HF energy since SCF may not be fully converged.
        ''' This block is modified to take precomputed 1e integrals
        '''
        s1e = getattr(mycc, '_s1e', None)
        h1e = getattr(mycc, '_h1e', None)
        vhf = getattr(mycc, '_vhf', None)
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        if vhf is None: vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm, h1e=h1e, s1e=s1e)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf, h1e=h1e)
        nocc = self.nocc = mycc.nocc
        self.mol = mycc.mol

        # Note self.mo_energy can be different to fock.diagonal().
        # self.mo_energy is used in the initial guess function (to generate
        # MP2 amplitudes) and CCSD update_amps preconditioner.
        # fock.diagonal() should only be used to compute the expectation value
        # of Slater determinants.
        mo_e = self.mo_energy = self.fock.diagonal().real
        try:
            gap = abs(mo_e[:nocc,None] - mo_e[None,nocc:]).min()
            if gap < 1e-5:
                logger.warn(mycc, 'HOMO-LUMO gap %s too small for CCSD.\n'
                            'CCSD may be difficult to converge. Increasing '
                            'CCSD Attribute level_shift may improve '
                            'convergence.', gap)
        except ValueError:  # gap.size == 0
            pass
        return self
def ccsd_ao2mo(self, mo_coeff=None):
    # Pseudo code how eris are implemented:
    # nocc = self.nocc
    # nmo = self.nmo
    # nvir = nmo - nocc
    # eris = _ChemistsERIs()
    # eri = ao2mo.incore.full(self._scf._eri, mo_coeff)
    # eri = ao2mo.restore(1, eri, nmo)
    # eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
    # eris.ovoo = eri[:nocc,nocc:,:nocc,:nocc].copy()
    # eris.ovvo = eri[nocc:,:nocc,nocc:,:nocc].copy()
    # eris.ovov = eri[nocc:,:nocc,:nocc,nocc:].copy()
    # eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
    # ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
    # eris.ovvv = lib.pack_tril(ovvv.reshape(-1,nvir,nvir))
    # eris.vvvv = ao2mo.restore(4, eri[nocc:,nocc:,nocc:,nocc:], nvir)
    # eris.fock = np.diag(self._scf.mo_energy)
    # return eris

    nmo = self.nmo
    nao = self.mo_coeff.shape[0]
    nmo_pair = nmo * (nmo+1) // 2
    nao_pair = nao * (nao+1) // 2
    mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (self._scf._eri is not None and
        (mem_incore+mem_now < self.max_memory or self.incore_complete)):
        return _make_eris_incore(self, mo_coeff)

    elif getattr(self._scf, 'with_df', None):
        logger.warn(self, 'CCSD detected DF being used in the HF object. '
                    'MO integrals are computed based on the DF 3-index tensors.\n'
                    'It\'s recommended to use dfccsd.CCSD for the '
                    'DF-CCSD calculations')
        return _make_df_eris_outcore(self, mo_coeff)

    else:
        raise NotImplementedError   # should never happen
        return _make_eris_outcore(self, mo_coeff)
def _make_eris_incore(mycc, mo_coeff=None):
    from pyscf import ao2mo

    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    #:eri1 = ao2mo.restore(1, eri1, nmo)
    #:eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    #:eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    #:eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    #:eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    #:eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    #:ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    #:eris.ovvv = lib.pack_tril(ovvv.reshape(-1,nvir,nvir)).reshape(nocc,nvir,-1)
    #:eris.vvvv = ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)

    if eri1.ndim == 4:
        eri1 = ao2mo.restore(4, eri1, nmo)

    nvir_pair = nvir * (nvir+1) // 2
    eris.oooo = np.empty((nocc,nocc,nocc,nocc))
    eris.ovoo = np.empty((nocc,nvir,nocc,nocc))
    eris.ovvo = np.empty((nocc,nvir,nvir,nocc))
    eris.ovov = np.empty((nocc,nvir,nocc,nvir))
    eris.ovvv = np.empty((nocc,nvir,nvir_pair))
    eris.vvvv = np.empty((nvir_pair,nvir_pair))

    ij = 0
    outbuf = np.empty((nmo,nmo,nmo))
    oovv = np.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        for j in range(i+1):
            eris.oooo[i,j] = eris.oooo[j,i] = buf[j,:nocc,:nocc]
            oovv[i,j] = oovv[j,i] = buf[j,nocc:,nocc:]
        ij += i + 1
    eris.oovv = oovv
    oovv = None

    ij1 = 0
    for i in range(nocc,nmo):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        eris.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
        eris.ovvo[:,i-nocc] = buf[:nocc,nocc:,:nocc]
        eris.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
        eris.ovvv[:,i-nocc] = lib.pack_tril(buf[:nocc,nocc:,nocc:])
        dij = i - nocc + 1
        lib.pack_tril(buf[nocc:i+1,nocc:,nocc:],
                      out=eris.vvvv[ij1:ij1+dij])
        ij += i + 1
        ij1 += dij
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris
def _make_df_eris_outcore(mycc, mo_coeff=None):
    from pyscf.ao2mo import _ao2mo

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mo_coeff = np.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2

    naux = mycc._scf.with_df.get_naoaux()
    Loo = np.empty((naux,nocc,nocc))
    Lov = np.empty((naux,nocc,nvir))
    Lvo = np.empty((naux,nvir,nocc))
    Lvv = np.empty((naux,nvir_pair))
    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    p1 = 0
    for eri1 in mycc._scf.with_df.loop():
        Lpq = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvo[p0:p1] = Lpq[:,nocc:,:nocc]
        Lvv[p0:p1] = lib.pack_tril(Lpq[:,nocc:,nocc:].reshape(-1,nvir,nvir))
    
#    import pdb;pdb.set_trace() 
    Loo = Loo.reshape(naux,nocc*nocc)
    Lov = Lov.reshape(naux,nocc*nvir)
    Lvo = Lvo.reshape(naux,nocc*nvir)

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8')
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir_pair,nvir_pair), 'f8')
    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    eris.oovv[:] = lib.unpack_tril(lib.ddot(Loo.T, Lvv)).reshape(nocc,nocc,nvir,nvir)
    eris.ovvo[:] = lib.ddot(Lov.T, Lvo).reshape(nocc,nvir,nvir,nocc)
    eris.ovov[:] = lib.ddot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovvv[:] = lib.ddot(Lov.T, Lvv).reshape(nocc,nvir,nvir_pair)
    eris.vvvv[:] = lib.ddot(Lvv.T, Lvv)
    #import pdb
    #pdb.set_trace()
    log.timer('CCSD integral transformation', *cput0)
    return eris
''' End of modification of PySCF's CCSD class
'''

''' impurity solver for LNO-based CCSD/CCSD_T
'''
def impurity_solve(mf, mo_coeff, lo_coeff, ccsd_t=False, eris=None, frozen=None,
                   log=None, verbose_imp=0):
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

    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]
    #import pdb;pdb.set_trace()
    nlo = lo_coeff.shape[1]
    s1e = mf.get_ovlp() if eris is None else eris.s1e
    prjlo = fdot(lo_coeff.T, s1e, orbactocc)

    log.debug('    impsol:  %d LOs  %d/%d MOs  %d occ  %d vir',
              nlo, nactocc+nactvir, nmo, nactocc, nactvir)

    # solve impurity problem
    from pyscf.cc import CCSD
    mcc = CCSD(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=verbose_imp)
    mcc.ao2mo = ccsd_ao2mo.__get__(mcc, mcc.__class__)
    mcc._s1e = s1e
    if eris is not None:
        mcc._h1e = eris.h1e
        mcc._vhf = eris.vhf
    imp_eris = mcc.ao2mo()
    # vhf is assumed to be computed with exxdiv=None and imp_eris.mo_energy is not
    # exxdiv-corrected. We correct it for MP2 energy if mf.exxdiv is 'ewald'.
    # FIXME: Should we correct it for other exxdiv options (e.g., 'vcut_sph')?
    if hasattr(mf, 'exxdiv') and mf.exxdiv == 'ewald':  # PBC HF object
        from pyscf.pbc.cc.ccsd import _adjust_occ
        from pyscf.pbc import tools
        madelung = tools.madelung(mf.cell, mf.kpt)
        imp_eris.mo_energy = _adjust_occ(imp_eris.mo_energy, imp_eris.nocc,
                                         -madelung)
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
    cput1 = log.timer_debug1('imp sol - mp2 ene', *cput1)
    # CCSD fragment energy
    t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
    cput1 = log.timer_debug1('imp sol - cc  amp', *cput1)
    t2 += einsum('ia,jb->ijab',t1,t1)
    elcorr_cc = get_fragment_energy(oovv, t2, prjlo)
    cput1 = log.timer_debug1('imp sol - cc  ene', *cput1)
    if ccsd_t:
        from ad_afqmc.lno.cc.ccsd_t import kernel as CCSD_T
        t2 -= einsum('ia,jb->ijab',t1,t1)   # restore t2
        elcorr_cc_t = CCSD_T(mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
        cput1 = log.timer_debug1('imp sol - cc  (T)', *cput1)
    else:
        elcorr_cc_t = 0.

    frag_msg = '  '.join([f'E_corr(MP2) = {elcorr_pt2:.15g}',
                          f'E_corr(CCSD) = {elcorr_cc:.15g}',
                          f'E_corr(CCSD(T)) = {elcorr_cc_t:.15g}'])

    t1 = t2 = oovv = imp_eris = mcc = None

    return frag_msg, (elcorr_pt2, elcorr_cc, elcorr_cc_t)

def get_fragment_energy(oovv, t2, prj):
    m = fdot(prj.T, prj)
    return einsum('ijab,kjab,ik->',t2,2*oovv-oovv.transpose(0,1,3,2),m)


class LNOCCSD(LNO):

    def __init__(self, mf, thresh=1e-6, frozen=None):

        LNO.__init__(self, mf, thresh=thresh, frozen=frozen)

        self.efrag_cc = None
        self.efrag_pt2 = None
        self.efrag_cc_t = None
        self.ccsd_t = False
        self.maxError = 0.
        self.runfrags = []

    def dump_flags(self, verbose=None):
        LNO.dump_flags(self, verbose=verbose)
        return self

    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris=None, frozen=None, log=None,chol_vecs=None,can_orbfrag=None):
        return impurity_solve(mf, mo_coeff, lo_coeff, eris=eris, frozen=frozen, log=log,
                              verbose_imp=self.verbose_imp, ccsd_t=self.ccsd_t)

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
        logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
                    'LNOMP2', self.e_tot_pt2, self.e_corr_pt2)
        logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
                    'LNOCCSD', self.e_tot, self.e_corr)
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
    def e_tot_ccsd_t(self):
        return self.e_corr_ccsd_t + self._scf.e_tot

    @property
    def e_tot_pt2(self):
        return self.e_corr_pt2 + self._scf.e_tot

    def e_corr_pt2corrected(self, ept2):
        #print(self.e_corr,self.e_corr_pt2,ept2) 
        #import pdb
        #pdb.set_trace()
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


class LNOCCSD_T(LNOCCSD):
    def __init__(self, mf, thresh=1e-6, frozen=None):
        LNOCCSD.__init__(self, mf, thresh=thresh, frozen=frozen)
        self.ccsd_t = True


if __name__ == '__main__':
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
# canonical
    mmp = mp.MP2(mf, frozen=frozen)
    mmp.kernel()

    mcc = cc.CCSD(mf, frozen=frozen).set(verbose=5)
    eris = mcc.ao2mo()
    mcc.kernel(eris=eris)

    from pyscf.cc.ccsd_t import kernel as CCSD_T
    eccsd_t = CCSD_T(mcc, eris)

# LNO
    for thresh in [1e-4]:
        mfcc = LNOCCSD(mf, thresh=thresh, frozen=frozen).set(verbose=5)
        mfcc.thresh_occ = 1e-3
        mfcc.thresh_vir = thresh
        mfcc.lo_type = 'pm'
        mfcc.no_type = 'cim'
        mfcc.frag_lolist = '1o'
        mfcc.frag_atmlist = None 
        mfcc.ccsd_t = True
        mfcc.force_outcore_ao2mo = True
        mfcc.frag_atmlist = tools.autofrag(mol, H2heavy=True)
        mfcc.kernel()
        ecc = mfcc.e_corr
        ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
       
      
        ecc_t = mfcc.e_corr_ccsd_t
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
