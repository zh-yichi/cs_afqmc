import sys
import os
import numpy as np
from functools import reduce

from pyscf import mp
from pyscf.lib import logger
from pyscf import lib

_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
einsum = lib.einsum

THRESH_INTERNAL = 1e-10


r''' TODO's
[x] Fast `make_fpno1` for non-CIM mode
[x] Combine kernels for CIM and non-CIM
[x] Support pbc.scf.RHF (i.e., Gamma-point PBC SCF)
[ ] LO pop analysis (warning if delocalized LOs are found)
[ ] chkfile / restart
[x] Refactor eris
[x] IBO
[ ] kmf
'''


def kernel(mfcc, orbloc, frag_lolist, no_type, eris=None, frag_nonvlist=None):
    r''' Kernel function for LNO-based methods.

    Args:
        orbloc (np.ndarray):
            Column vectors are the AO coefficients for a set of local(ized) orbitals.
            These LOs must span at least the occupied space but can span none, part, or
            full of the virtual space. Thus, orbloc.shape[1] >= nmo.
        frag_lolist (list of list):
            Fragment definition in terms of the LOs specified by 'orbloc'. E.g.,
                [[0,1,2],[3,5],[4],[6,7,8,9]...]
            means
                fragment 1 consists of LO 0, 1, and 2,
                fragment 2 consists of LO 3 and 5,
                fragment 3 consists of LO 4,
                fragment 4 consists of LO 6, 7, 8, and 9,
                ...
        no_type (str of two chars):
            A string of two chars specifying the type of natural orbitals (NOs) used to
            compress the occupied and virtual space, respectively. Each string can be one
            of the following (lower/upper cases for canonical/local orbitals):
                - 'r' for 'restricted' : iAkB,jAkB->ij and IaJc,IbJc->ab
                - 'e' for 'extended'   : iAkb,jAkb->ij and Iajc,Ibjc->ab
                - 'i' for 'inverted'   : iaKb,jaKb->ij and iajC,ibjC->ab
            Note that 'r' and 'e' for occ and 'i' for vir must not be used when LOs
            span only the occ space (e.g., those from localizing occupied MOs).

            Some examples that correspond to known methods in literature:
                - 'rr': used by Nusspickel and Booth in their extension of density matrix
                embedding theory (DMET); see arXiv:2107.04916v3.
                - 'ie': used by Rolik and Kallay in their extension of the cluster-in-
                molecule (CIM) method; see J. Chem. Phys. 135, 104111 (2011).
    '''
    nfrag = len(frag_lolist)
    if(type(mfcc).__name__ =='LNOAFQMC'): canonicalize = False;mfcc.maxError = mfcc.maxError/np.sqrt(nfrag)
    else: canonicalize = True 
    log = logger.Logger(mfcc.stdout, mfcc.verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())

    # quick sanity check for no_type (more in 'make_fpno1')
    if not (no_type[0] in 'rei' and no_type[1] in 'rei'):
        log.warn('Input no_type "%s" is invalid.', no_type)
        raise ValueError

    if frag_nonvlist is None: frag_nonvlist = [[None,None]] * nfrag

    if eris is None: eris = mfcc.ao2mo()

    cput1 = (logger.process_clock(), logger.perf_counter())
## Loop over fragment
    frag_res = [None] * nfrag

    for ifrag in range(0,nfrag):
    # local basis for internal space
        if(len(mfcc.runfrags)>0):
            if(ifrag not in mfcc.runfrags):frag_res[ifrag] = (0,0,0);continue
        fraglo = frag_lolist[ifrag]
        orbfragloc = orbloc[:,fraglo]
        frag_target_nocc, frag_target_nvir = frag_nonvlist[ifrag]
        frag_msg, frag_res[ifrag] = kernel_1frag(mfcc, eris, orbfragloc, no_type,
                                                 frag_target_nocc=frag_target_nocc,
                                                 frag_target_nvir=frag_target_nvir,canonicalize=canonicalize,ifrag=ifrag)
        cput1 = log.timer('Fragment %d'%(ifrag+1)+' '*(8-len(str(ifrag+1))), *cput1)
        log.info('Fragment %d  %s', ifrag+1, frag_msg)
        
        if(type(mfcc).__name__ =='LNOAFQMC'):
            os.system(f"mv afqmc.out afqmc_{ifrag}.out")
        
    classname = mfcc.__class__.__name__
    cput0 = log.timer(classname+' '*(17-len(classname)), *cput0)

    return frag_res

def kernel_1frag(mfcc, eris, orbfragloc, no_type, **kwargs):
    '''
    kwargs:
        frag_target_nocc/frag_target_nvir (int):
            If provided, the number of occ/vir NOs will be controlled.
    '''
    log = logger.Logger(mfcc.stdout, mfcc.verbose)
    cput1 = (logger.process_clock(), logger.perf_counter())

    mf = mfcc._scf
    frozen_mask = mfcc.get_frozen_mask()
    thresh_pno = [mfcc.thresh_occ, mfcc.thresh_vir]
    # ccsd_t = mfcc.ccsd_t
    frag_target_nocc = kwargs.get('frag_target_nocc', None)
    frag_target_nvir = kwargs.get('frag_target_nvir', None)
    canonicalize = kwargs.get('canonicalize',True)
    # chol_vecs = kwargs.get('chol_vecs',None)
    # ifrag = kwargs.get('ifrag',-1)
# make fpno
    frzfrag, orbfrag, can_orbfrag = make_fpno1(mfcc, eris, orbfragloc, no_type,
                                               THRESH_INTERNAL, thresh_pno,
                                               frozen_mask=frozen_mask,
                                               frag_target_nocc=frag_target_nocc,
                                               frag_target_nvir=frag_target_nvir,
                                               canonicalize=canonicalize)
    cput1 = log.timer('make pno         ', *cput1)
    
# solve impurity
    if(canonicalize == True): frag_msg, frag_res = mfcc.impurity_solve(mf,orbfrag,orbfragloc,frozen=frzfrag,
                                                                       eris=eris,log=log)
    #else: frag_msg, frag_res = mfcc.impurity_solve(mf, orbfrag, orbfragloc,frozen=frzfrag, eris=eris, log=log,can_orbfrag=can_orbfrag,nblocks = mfcc.nblocks,seed = mfcc.seed,chol_cut = mfcc.chol_cut, cholesky_threshold = mfcc.cholesky_threshold)
    else : frag_msg, frag_res = mfcc.impurity_solve(mf,orbfrag,orbfragloc,frozen=frzfrag,
                                                    eris=eris,log=log,can_orbfrag=can_orbfrag)
    cput1 = log.timer('imp sol          ', *cput1)
    return frag_msg, frag_res

def make_fpno1(mfcc, eris, orbfragloc, no_type, thresh_internal, thresh_external,
               frozen_mask=None, frag_target_nocc=None, frag_target_nvir=None,canonicalize=True):
    log = logger.Logger(mfcc.stdout, mfcc.verbose)
    mf = mfcc._scf
    nocc = np.count_nonzero(mf.mo_occ>1e-10)
    nmo = mf.mo_occ.size
    orbocc0, orbocc1, orbvir1, orbvir0 = mfcc.split_mo() # frz_occ, act_occ, act_vir, frz_vir
    moeocc0, moeocc1, moevir1, moevir0 = mfcc.split_moe() # split energy
    # nocc0, nocc1, nvir1, nvir0 = [m.size for m in [moeocc0,moeocc1,
    #                                                moevir1,moevir0]]
    # nlo = orbfragloc.shape[1]
    s1e = eris.s1e # if eris.s1e is None else mf.get_ovlp()
    fock = eris.fock # if eris.fock is None else mf.get_fock()
    Lov = eris.Lov
    # chosen loc_orb overlap with act_vir
    lovir = abs(fdot(orbfragloc.T, s1e, orbvir1)).max() > 1e-10

    if isinstance(thresh_external, float):
        thresh_ext_occ = thresh_ext_vir = thresh_external
    else:
        thresh_ext_occ, thresh_ext_vir  = thresh_external

    # sanity check for no_type:
    if not lovir and no_type[0] != 'i':
        log.warn('Input LOs span only occ but input no_type[0] is not "i".')
        raise ValueError
    if not lovir and no_type[1] == 'i':
        log.warn('Input LOs span only occ but input no_type[1] is "i".')
        raise ValueError

    # split active occ/vir into internal(1) and external(2)
    m = fdot(orbfragloc.T, s1e, orbocc1) # overlap with all loc act_occs
    uocc1, uocc2 = projection_construction(m, thresh_internal)
    moefragocc1, orbfragocc1 = subspace_eigh(fock, fdot(orbocc1, uocc1))
    if lovir:
        m = fdot(orbfragloc.T, s1e, orbvir1)
        uvir1, uvir2 = projection_construction(m, thresh_internal)
        moefragvir1, orbfragvir1 = subspace_eigh(fock, fdot(orbvir1, uvir1))

    def moe_Ov(moefragocc):
        return (moefragocc[:,None] - moevir1).reshape(-1)
    def moe_oV(moefragvir):
        return (moeocc1[:,None] - moefragvir).reshape(-1)
    eov = moe_Ov(moeocc1)
    # Construct PT2 dm_vv
    if no_type[1] == 'r':   # OvOv: IaJc,IbJc->ab
        u = fdot(orbocc1.T, s1e, orbfragocc1)
        ovov = eris.get_OvOv(u)
        eia = ejb = moe_Ov(moefragocc1)
        e1_or_e2 = 'e1'
        swapidx = 'ab'
    elif no_type[1] == 'e': # Ovov: Iajc,Ibjc->ab
        u = fdot(orbocc1.T, s1e, orbfragocc1)
        ovov = eris.get_Ovov(u)
        eia = moe_Ov(moefragocc1)
        Ljb = Lov
        ejb = eov
        e1_or_e2 = 'e1'
        swapidx = 'ab'
    else:                   # oVov: iCja,iCjb->ab
        u = fdot(orbvir1.T, s1e, orbfragvir1)
        ovov = eris.get_oVov(u)
        eia = moe_oV(moefragvir1)
        Ljb = Lov
        ejb = eov
        e1_or_e2 = 'e2'
        swapidx = 'ij'

    eiajb = (eia[:,None]+ejb).reshape(*ovov.shape)
    t2 = ovov / eiajb

    dmvv = make_rdm1_mp2(t2, 'vv', e1_or_e2, swapidx)
   
    if lovir:
        dmvv = fdot(uvir2.T, dmvv, uvir2)

    Lia = Ljb = ovov = eiajb = None
    # Construct PT2 dm_oo
    if no_type in ['ie','ei']: # ie/ei share same t2
        if no_type[0] == 'e':   # oVov: iAkb,jAkb->ij
            e1_or_e2 = 'e1'
            swapidx = 'ij'
        else:                   # Ovov: Kaib,Kajb->ij
            e1_or_e2 = 'e2'
            swapidx = 'ab'
    else:
        t2 = None

        if no_type[0] == 'r':   # oVoV: iAkB,jAkB->ij
            u = fdot(orbvir1.T, s1e, orbfragvir1)
            ovov = eris.get_oVoV(u)
            eia = ejb = moe_oV(moefragvir1)
            e1_or_e2 = 'e1'
            swapidx = 'ab'
        elif no_type[0] == 'e': # oVov: iAkb,jAkb->ij
            u = fdot(orbvir1.T, s1e, orbfragvir1)
            ovov = eris.get_oVov(u)
            eia = moe_oV(moefragvir1)
            Ljb = Lov
            ejb = eov
            e1_or_e2 = 'e1'
            swapidx = 'ij'
        else:                   # Ovov: Kaib,Kajb->ij
            u = fdot(orbocc1.T, s1e, orbfragocc1)
            ovov = eris.get_Ovov(u)
            eia = moe_Ov(moefragocc1)
            Ljb = Lov
            ejb = eov
            e1_or_e2 = 'e2'
            swapidx = 'ab'

        eiajb = (eia[:,None]+ejb).reshape(*ovov.shape)
        t2 = ovov / eiajb

        Lia = Ljb = ovov = eiajb = None

    dmoo = make_rdm1_mp2(t2, 'oo', e1_or_e2, swapidx)
    dmoo = fdot(uocc2.T, dmoo, uocc2)

    t2 = None
    # Compress external space by PNO
    
    if frag_target_nocc is not None: frag_target_nocc -= orbfragocc1.shape[1]
    orbfragocc2, orbfragocc0 = natorb_compression(dmoo, orbocc1, thresh_ext_occ,
                                                  uocc2, frag_target_nocc)
#    if (canonicalize): orbfragocc12 = subspace_eigh(fock, np.hstack([orbfragocc2, orbfragocc1]))[1]
#    else: orbfragocc12 = np.hstack([orbfragocc2, orbfragocc1])
    can_orbfragocc12 = subspace_eigh(fock, np.hstack([orbfragocc2, orbfragocc1]))[1]
    orbfragocc12 = np.hstack([orbfragocc2, orbfragocc1])
    if lovir:
        
        if frag_target_nvir is not None: frag_target_nvir -= orbfragvir1.shape[1]
        orbfragvir2, orbfragvir0 = natorb_compression(dmvv, orbvir1, thresh_ext_vir,
                                                      uvir2, frag_target_nvir)
        #if (canonicalize): orbfragvir12 = subspace_eigh(fock, np.hstack([orbfragvir2, orbfragvir1]))[1]
        #else: orbfragvir12 = np.hstack([orbfragvir2, orbfragvir1])
        can_orbfragvir12 = subspace_eigh(fock, np.hstack([orbfragvir2, orbfragvir1]))[1]
        orbfragvir12 = np.hstack([orbfragvir2, orbfragvir1])
    else: 
        orbfragvir2, orbfragvir0 = natorb_compression(dmvv, orbvir1, thresh_ext_vir,
                                                      None, frag_target_nvir)
        #if (canonicalize): orbfragvir12 = subspace_eigh(fock, orbfragvir2)[1]
        #else: orbfragvir12 = orbfragvir2
        can_orbfragvir12 = subspace_eigh(fock, orbfragvir2)[1]
        orbfragvir12 = orbfragvir2

    orbfrag = np.hstack([orbocc0, orbfragocc0, orbfragocc12,
                         orbfragvir12, orbfragvir0, orbvir0])
    can_orbfrag = np.hstack([orbocc0, orbfragocc0, can_orbfragocc12,
                        can_orbfragvir12, orbfragvir0, orbvir0])
    
    frzfrag = np.hstack([np.arange(orbocc0.shape[1]+orbfragocc0.shape[1]),
                         np.arange(nocc+orbfragvir12.shape[1],nmo)])

    #return frzfrag, orbfrag
    #import pdb;pdb.set_trace()
    if(canonicalize==True): return frzfrag, can_orbfrag, can_orbfrag 
    else: return frzfrag, orbfrag , can_orbfrag

def _matpow(A, p):
    e, u = np.linalg.eigh(A)
    return _fdot( u*e**p, u.T )

def get_iao(mol, mo, minao='minao', orth=True):
    from pyscf.lo.iao import iao as iao_kernel
    orbiao = iao_kernel(mol, mo, minao=minao)

    if orth:
        if hasattr(mol, 'pbc_intor'):   # cell
            s1e = mol.pbc_intor('int1e_ovlp')
        else:
            s1e = mol.intor('int1e_ovlp')
        ovlpiao = _fdot( orbiao.T, _fdot( s1e, orbiao ) )
        orbiao = _fdot( orbiao, _matpow(ovlpiao, -0.5) )

    return orbiao

def get_ibo(mol, mo, minao='minao'):
    from pyscf.lo.ibo import ibo as ibo_kernel
    orbibo = ibo_kernel(mol, mo, locmethod='ibo', minao=minao)
    return orbibo

def get_boys(mol, mo):
    from pyscf.lo.boys import Boys
    loc = Boys(mol, mo_coeff=mo).set(verbose=4)
    loc.max_stepsize = 0.01
    orbloc = loc.kernel()
    return orbloc

def get_pm(mol, mo):
    from pyscf.lo.pipek import PipekMezey as PM
    loc = PM(mol, mo_coeff=mo).set(verbose=0)
    orbloc = loc.kernel()
    return orbloc

def map_lo_to_frag(mol, orbloc, frag_atmlist, verbose=None):
    r''' Assign input LOs (assumed orthonormal) to fragments using the Mulliken charge.

    For each IAO 'i', a 1D array, [p_1, p_2, ... p_nfrag], is computed, where
        p_ifrag = \sum_{mu on fragment i} ( (s1e^{1/2}*orbloc)[mu,i] )**2.
    '''
    if verbose is None: verbose = mol.verbose
    log = logger.Logger(mol.stdout, verbose)

    if hasattr(mol, 'pbc_intor'):
        s1e = mol.pbc_intor('int1e_ovlp')
    else:
        s1e = mol.intor('int1e_ovlp')
    s1e_sqrt = _matpow(s1e, 0.5)
    plo_ao = _fdot(s1e_sqrt, orbloc)**2.
    aoslice_by_atom = mol.aoslice_nr_by_atom()
    aoind_by_frag = [np.concatenate([range(*aoslice_by_atom[atm][-2:])
                                     for atm in atmlist])
                     for atmlist in frag_atmlist]
    plo_frag = np.array([plo_ao[aoind].sum(axis=0)
                         for aoind in aoind_by_frag]).T
    lo_frag_map = plo_frag.argmax(axis=1)
    nlo, nfrag = plo_frag.shape
    for i in range(nlo):
        log.debug1('IAO %d is assigned to frag %d with charge %.2f',
                   i, lo_frag_map[i], plo_frag[i,lo_frag_map[i]])
        log.debug2('pop by frag:' + ' %.2f'*nfrag, *plo_frag[i])

    frag_lolist = [np.where(lo_frag_map==i)[0] for i in range(nfrag)]

    return frag_lolist

def projection_construction(M, thresh):
    r''' Given M_{mu,i} = <mu | i> the ovlp between two orthonormal basis, find
    the unitary rotation |j'> = u_ij |i> so that {|j'>} significantly ovlp with
    {|mu>}.
    '''
    n, m = M.shape
    e, u = np.linalg.eigh(fdot(M.T, M))
    mask = abs(e) > thresh
    return u[:,mask], u[:,~mask]

def subspace_eigh(fock, orb):
    f = fdot(orb.T, fock, orb)
    if orb.shape[1] == 1:
        moe = np.array([f[0,0]])
    else:
        moe, u = np.linalg.eigh(f)
        orb = fdot(orb, u)
    return moe, orb

def natorb_compression(dm, orb, thresh, prj=None, norb_target=None):
    e, u = np.linalg.eigh(dm)
#    import pdb;pdb.set_trace()
    if norb_target is None:
        idx = np.where(abs(e) > thresh)[0]
    elif isinstance(norb_target, (int,np.integer)):
        if norb_target < 0:
            raise ValueError('Target norb is negative: %d.' % norb_target)
        elif norb_target > e.size:
            raise ValueError('Target norb exceeds total number of orbs: %d > %d'
                             % (norb_target, e.size))
        order = e.argsort()[::-1]
        idx = order[:norb_target]
    else:
        raise TypeError('Input "norb_target" should be integer type.')
    idxc = np.array([i for i in range(e.size) if i not in idx])
    if prj is not None:
        orbx = fdot(orb, fdot(prj, u))
    else:
        orbx = fdot(orb, u)
    orb1x = sub_colspace(orbx, idx)
    orb0x = sub_colspace(orbx, idxc)
    return orb1x, orb0x

def sub_colspace(A, idx):
    if idx.size == 0:
        return np.zeros([A.shape[0],0])
    else:
        return A[:,idx]

def get_ovov(mf, orbo, orbv):
    no = orbo.shape[1]
    nv = orbv.shape[1]
    if mf._eri is not None:
        from pyscf import ao2mo
        ovov = ao2mo.incore.general(mf._eri,(orbo,orbv,
                                             orbo,orbv)).reshape(no,nv,no,nv)
    elif getattr(mf, 'with_df', None):
        from pyscf.ao2mo import _ao2mo
        mo = np.asarray(np.hstack([orbo,orbv]), order='F')
        ijslice = (0, no, no, no+nv)
        Lov = None
        ovov = 0.
        for Lmunu in mf.with_df.loop():
            Lov = _ao2mo.nr_e2(Lmunu, mo, ijslice, aosym='s2', out=Lov)
            ovov += Lov.T @ Lov
            Lmunu = None
        ovov = ovov.reshape(no,nv,no,nv)
    else:
        from pyscf import ao2mo
        mol = mf.mol
        ovov = ao2mo.general(mol,(orbo,orbv,orbo,orbv)).reshape(no,nv,no,nv)
    return ovov

def get_erimo(mf, orb1, orb2, orb3, orb4):
    n1, n2, n3, n4 = [orb.shape[1] for orb in [orb1,orb2,orb3,orb4]]
    if mf._eri is not None:
        from pyscf import ao2mo
        erimo = ao2mo.incore.general(mf._eri,(orb1,orb2,orb3,orb4))
    elif getattr(mf, 'with_df', None):
        from pyscf.ao2mo import _ao2mo
        mo12 = np.asarray(np.hstack([orb1,orb2]), order='F')
        ijslice12 = (0, n1, n1, n1+n2)
        mo34 = np.asarray(np.hstack([orb3,orb4]), order='F')
        ijslice34 = (0, n3, n3, n3+n4)
        erimo = 0.
        for Lmunu in mf.with_df.loop():
            Lov12 = _ao2mo.nr_e2(Lmunu, mo12, ijslice12, aosym='s2')
            Lov34 = _ao2mo.nr_e2(Lmunu, mo34, ijslice34, aosym='s2')
            erimo += Lov12.T @ Lov34
            Lmunu = None
    else:
        from pyscf import ao2mo
        mol = mf.mol
        erimo = ao2mo.general(mol,(orb1,orb2,orb3,orb4))
    erimo = erimo.reshape(n1,n2,n3,n4)
    return erimo

def make_rdm1_mp2(t2, kind, e1_or_e2, swapidx):
    r''' Calculate MP2 rdm1 from T2.

    Args:
        t2 (np.ndarray):
            In 'ovov' order.
        kind (str):
            'oo' for oo-block; 'vv' for vv-block
        e1_or_e2 (str):
            Which electron are the free indices on?
            'e1': iakb,jakb -> ij; iajc,ibjc -> ab
            'e2': kaib,kajb -> ij; icja,icjb -> ab
        swapidx (str):
            How is the exchange term handled in einsum?
            'ij': iajb --> jaib
            'ab': iajb --> ibja
    '''
    if kind not in ['oo','vv']:
        raise ValueError('kind must be "oo" or "vv".')
    if e1_or_e2 not in ['e1','e2']:
        raise ValueError('e1_or_e2 must be "e1" or "e2".')
    if swapidx not in ['ij','ab']:
        raise ValueError('swapidx must be "ij" or "ab".')

    def swapped(s, swapidx):
        assert(len(s) == 4)
        order = [2,1,0,3] if swapidx == 'ij' else [0,3,2,1]
        return ''.join([s[i] for i in order])

    if kind == 'oo':
        if e1_or_e2 == 'e1':
            ids0 = 'iakb'
            ids1 = 'jakb'
        else:
            ids0 = 'kaib'
            ids1 = 'kajb'
        ids2 = 'ij'
    else:
        if e1_or_e2 == 'e1':
            ids0 = 'iajc'
            ids1 = 'ibjc'
        else:
            ids0 = 'icja'
            ids1 = 'icjb'
        ids2 = 'ab'
    ids0x = swapped(ids0, swapidx)
    ids1x = swapped(ids1, swapidx)

    merge_ids = lambda s0,s1,s2: '->'.join([','.join([s0,s1]),s2])
    dm = (einsum(merge_ids(ids0 , ids1 , ids2), t2, t2)*2 -
          einsum(merge_ids(ids0 , ids1x, ids2), t2, t2)   -
          einsum(merge_ids(ids0x, ids1 , ids2), t2, t2)   +
          einsum(merge_ids(ids0x, ids1x, ids2), t2, t2)*2) * 0.5

    return dm

def mo_splitter(maskact, maskocc, kind='mask'):
    ''' Split MO indices into
        - frozen occupieds
        - active occupieds
        - active virtuals
        - frozen virtuals

    Args:
        maskact (array-like, bool type):
            An array of length nmo with bool elements. True means an MO is active.
        maskocc (array-like, bool type):
            An array of length nmo with bool elements. True means an MO is occupied.
        kind (str):
            Determine the return type.
            'mask'  : return masks each of length nmo
            'index' : return index arrays
            'idx'   : same as 'index'

    Return:
        See the description for input arg 'kind' above.
    '''
    maskfrzocc = ~maskact &  maskocc
    maskactocc =  maskact &  maskocc
    maskactvir =  maskact & ~maskocc
    maskfrzvir = ~maskact & ~maskocc
    if kind == 'mask':
        return maskfrzocc, maskactocc, maskactvir, maskfrzvir
    elif kind in ['index','idx']:
        return [np.where(m)[0] for m in [maskfrzocc, maskactocc,
                                         maskactvir, maskfrzvir]]
    else:
        raise ValueError("'kind' must be 'mask' or 'index'(='idx').")


class LNO(lib.StreamObject):

    r''' Base class for LNO-based methods

    This base class provides common functions for constructing LNO subspace.
    Specific LNO-based methods (e.g., LNO-CCSD, LNO-CCSD(T)) can be implemented as
    derived classes from this base class with appropriately defined method
    `impurity_solve`.

    Input:
        mf (PySCF SCF object):
            Mean-field object.
        thresh (float):
            Cutoff for PNO selection.
        frozen (int or list):
            Same as the `frozen` attr in MP2/CCSD etc. modules.
    '''

    def __init__(self, mf, thresh=1e-6, frozen=None):

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        self.thresh = thresh
        self.thresh_occ = thresh
        self.thresh_vir = thresh
        self.canonicalize = True
        self.chol_vecs = None

        # non-exposed argument
        self.lo_type = 'pm'
        self.no_type = 'cim' # same as 'ie'
        self.orbloc = None
        self.frag_atmlist = None
        self.frag_lolist = None
        self.frag_wghtlist = None
        self.frag_nonvlist = None
        self.force_outcore_ao2mo = False
        self.verbose_imp = 0 # allow separate verbose level for `impurity_solve`

        # Not input options
        self.nfrag = None
        self._nmo = None
        self._nocc = None
        self.mo_occ = self._scf.mo_occ

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('thresh_occ = %e', self.thresh_occ)
        log.info('thresh_vir = %e', self.thresh_vir)
        log.info('force_outcore_ao2mo = %r', self.force_outcore_ao2mo)
        return self

    def mo_splitter(self, kind='mask'):
        r''' Return index arrays that split MOs into
            - frozen occupieds
            - active occupieds
            - active virtuals
            - frozen virtuals

        Args:
            kind (str):
                'mask'  : return masks each of length nmo
                'index' : return index arrays
                'idx'   : same as 'index'
        '''
        maskact = self.get_frozen_mask()
        maskocc = self._scf.mo_occ > 1e-10
        return mo_splitter(maskact, maskocc, kind=kind)

    def split_mo(self):
        r''' Return the four components of MOs specified in :func:`mo_splitter`
        '''
        mo = self._scf.mo_coeff
        masks = self.mo_splitter()
        return [mo[:,m] for m in masks]

    def split_moe(self):
        moe = self._scf.mo_energy
        masks = self.mo_splitter()
        return [moe[m] for m in masks]

    def get_lo(self, mo=None, lo_type=None):
        r''' Get localized orbitals as a basis for internal space. Return the
        coeff matrix in AO basis.
        '''
        if lo_type is None: lo_type = 'iao'
        logger.info(self, 'lo_type = %s', lo_type)
        if lo_type.lower() == 'iao':
            if mo is None: mo = np.hstack(self.split_mo()[:2])  # occ
            orbloc = get_iao(self.mol, mo, orth=True)
        elif lo_type.lower() == 'ibo':
            if mo is None: mo = np.hstack(self.split_mo()[:2])  # occ
            orbloc = get_ibo(self.mol, mo)
        elif lo_type.lower() == 'boys':
            if mo is None: mo = self.split_mo()[1]  # active occ
            orbloc = get_boys(self.mol, mo)
        elif lo_type.lower() == 'pm':
            if mo is None: mo = self.split_mo()[1]  # active occ
            orbloc = get_pm(self.mol, mo)
        else:
            raise NotImplementedError

        return orbloc

    def kernel(self, eris=None, orbloc=None, lo_type=None, no_type=None,
               frag_lolist=None, frag_atmlist=None, frag_wghtlist=None,
               frag_nonvlist=None, canonicalize=True):
        r'''
        Args:
            orbloc (matrix, nao * nlo):
                AO coefficient matrix for the LO basis.
            lo_type (str):
                If `orbloc` is not provided, call self.get_lo(lo_type) to generate LOs.
                Options include 'iao', 'boys', and 'pm'. The default is 'pm'.
            no_type (str):
                A string of two chars specifying the type of natural orbitals (NOs)
                used to compress the occupied and virtual space, respectively.
                Each string can be (lower/upper cases for canonical/local orbitals):
                    - 'r' for 'restricted' : iAkB,jAkB->ij and IaJc,IbJc->ab
                    - 'e' for 'extended'   : iAkb,jAkb->ij and Iajc,Ibjc->ab
                    - 'i' for 'inverted'   : iaKb,jaKb->ij and iajC,ibjC->ab
                Some special shortcuts:
                    'edmet' <=> 'rr' : Nusspickel and Booth; see arXiv:2107.04916v3.
                    'cim'   <=> 'ie' : Rolik and Kallay; see JCP 135, 104111 (2011).
                The default is 'cim'.

                Note that 'r' and 'e' for occ and 'i' for vir must *not* be used with
                LOs spanning only the occ space (e.g., lo_type = 'boys' or 'pm').
            frag_lolist (list of list):
                Fragment assignment in terms of LO index. E.g.,
                    [[0,1],[2],[3,4,5],...]
                means
                    fragment 1 = LO 0 + LO 1,
                    fragment 2 = LO 2
                    fragment 3 = LO 3 + LO 4 + LO 5,
                    ...
                If not provided, it is generated from `frag_atmlist` by calling
                :func:`map_lo_to_frag`.
            frag_atmlist (list of list):
                Fragment assignment in terms of atoms.
                Same syntax as `frag_lolist` but each number referring to the
                atom number (in the order of user input 'atom' for constructing
                the Mole object). If not provided, :func:`autofrag` from
                lno/tools/tools.py is called, which generates single-atom
                fragments (hydrogen is grouped with the closed heavy atom).
            frag_wghtlist (array-like):
                An array of the same length as the frag_atmlist/frag_lolist. The total
                energy is evaluated as a weighted-sum from each fragment with the weights
                specified by this array. If not provided, np.ones(nfrag) will be used.
            frag_nonvlist (ndarray-like of shape (nfrag, 2)):
                A list of size-2 arrays each specifying the number of occupied and
                virtual orbitals for a fragment. Note that this includes both internal
                (usually LOs projected to occ and vir space) and external (important
                NOs) orbitals.
        '''
        self.dump_flags()

        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())

        if orbloc is None: orbloc = self.orbloc
        if lo_type is None: lo_type = self.lo_type
        if no_type is None: no_type = self.no_type
        if frag_atmlist is None: frag_atmlist = self.frag_atmlist
        if frag_lolist is None: frag_lolist = self.frag_lolist
        if frag_wghtlist is None: frag_wghtlist = self.frag_wghtlist

        s1e = self._scf.get_ovlp()

        # NO type
        if no_type == 'cim':
            no_type = 'ie'
        elif no_type == 'edmet':
            no_type = 'rr'
        log.info('no_type = %s', no_type)

        # LO construction
        if orbloc is None:
            log.info('Constructing LOs')
            orbloc = self.get_lo(lo_type=lo_type)
        else:
            log.info('Using user input LOs')
#        import QMCUtils,os
#        mo = self._scf.mo_coeff
#        mo[:,:10] = orbloc
#        for i in range(0,8):
#        	QMCUtils.run_afqmc_lno_mf(self._scf,mo_coeff = mo,nblocks=800,norb_frozen=2,orbitalE=i)
#        	os.system(f'mv afqmc.dat afqmc{i}.dat;mv blocking.out blocking{i}.out')
#        exit(0)
        # check 1: Span(LO) >= Span(occ)
        #import pdb;pdb.set_trace()
        #self.active_space_mo = orbloc[:,self.active_space]
        orbactocc = self.split_mo()[1]
        m = fdot(orbloc.T, s1e, orbactocc)
        lospanerr = abs(fdot(m.T, m) - np.eye(m.shape[1])).max()
        if lospanerr > 1e-10:
            log.error('LOs do not fully span the occupied space! '
                      'Max|<occ|LO><LO|occ>| = %e', lospanerr)
            raise RuntimeError

        # check 2: Span(LO) == Span(occ)
        occspanerr = abs(fdot(m, m.T) - np.eye(m.shape[0])).max()
        if occspanerr < 1e-10:
            log.info('LOs span exactly the occupied space.')
            if no_type not in ['ir','ie']:
                log.error('"no_type" must be "ir" or "ie".')
                raise ValueError
        else:
            log.info('LOs span occupied space plus some virtual space.')

        # LO assignment to fragments
        if frag_lolist is None:
            if frag_atmlist is None:
                log.info('Grouping LOs by single-atom fragments')
                from ad_afqmc.lno.tools import autofrag
                frag_atmlist = autofrag(self.mol)
            else:
                log.info('Grouping LOs by user input atom-based fragments')
            frag_lolist = map_lo_to_frag(self.mol, orbloc, frag_atmlist,
                                         verbose=self.verbose)
        elif frag_lolist == '1o':
            log.info('Using single-LO fragment')
            frag_lolist = [[i] for i in range(orbloc.shape[1])]
        else:
            log.info('Using user input LO-fragment assignment')
        nfrag = len(frag_lolist)

        # frag weights
        if frag_wghtlist is None:
            frag_wghtlist = np.ones(nfrag)
        elif isinstance(frag_wghtlist, (list,np.ndarray)):
            if len(frag_wghtlist) != nfrag:
                log.error('Input frag_wghtlist has wrong length (expecting %d; '
                          'got %d).', nfrag, len(frag_wghtlist))
                raise ValueError
            frag_wghtlist = np.asarray(frag_wghtlist)
        else:
            log.error('Input frag_wghtlist has wrong data type (expecting '
                      'array-like; got %s)', type(frag_wghtlist))
            raise ValueError

        if frag_nonvlist is None: frag_nonvlist = self.frag_nonvlist

        # dump info
        log.info('nfrag = %d  nlo = %d', nfrag, orbloc.shape[1])
        log.info('frag_atmlist = %s', frag_atmlist)
        log.info('frag_lolist = %s', frag_lolist)
        log.info('frag_wghtlist = %s', frag_wghtlist)
        log.info('frag_nonvlist = %s', frag_nonvlist)

        log.timer('LO and fragment  ', *cput0)
        #if(self.multislater):
        #  #import pdb;pdb.set_trace()
        #  mo = self._scf.mo_coeff
        #  ovlp_matrix = self._scf.get_ovlp(self._scf.mol)
        #  active_space = [8,9]
        #  mo_1 = mo[:,active_space[0]]
        #  overlap = mo_1 @ ovlp_matrix @ mo

        frag_res = kernel(self, orbloc, frag_lolist, no_type,
                          eris=eris, frag_nonvlist=frag_nonvlist)

        self._post_proc(frag_res, frag_wghtlist)

        self._finalize()

        return self.e_corr

    def ao2mo(self):
        ''' Lov for df or ovov for non-df.
        '''
        log = logger.Logger(self.stdout, self.verbose)
        cput0 = (logger.process_clock(), logger.perf_counter())
        orbocc, orbvir = self.split_mo()[1:3]
        nocc = orbocc.shape[1]
        nvir = orbvir.shape[1]
        mf = self._scf
        # FIXME: more accurate mem estimate
        mem_now = self.max_memory - lib.current_memory()[0]
        mem_incore0 = (nocc*nvir)**2*8/1024**2.
        mem_incore = mem_incore0 * 3 # 3 for tIajb, eIajb etc.
        if getattr(mf, 'with_df', None):
            naux = mf.with_df.get_naoaux()
            mem_df0 = nocc*nvir*naux*8/1024**2.
            mem_df = mem_df0 * 2 # 2 for LOv etc.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            if (mem_df < mem_now) and not self.force_outcore_ao2mo:
                eris = _make_df_eris_incore(self)
            else:
                eris = _make_df_eris_outcore(self)
        elif mf._eri is not None and mem_incore < mem_now:
            eris = _make_eris_incore(self)
        else:
            eris = _make_eris_outcore(self)
        cput1 = log.timer('Integral xform   ', *cput0)

        return eris

    get_frozen_mask = mp.mp2.get_frozen_mask
    get_nocc = mp.mp2.get_nocc
    get_nmo = mp.mp2.get_nmo

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

    ''' The following methods need to be implemented for derived LNO classes.
    '''
    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris=None, frozen=None, log=None):
        log = logger.new_logger(self)
        log.error('You are calling the base LNO class! Please call the method-specific '
                  'LNO classes.')
        raise NotImplementedError

    def _post_proc(self, frag_res, frag_wghtlist):
        pass

    def _finalize(self):
        pass

    @property
    def e_corr(self):
        pass


class _LNOERIS(lib.StreamObject):

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None

        self.h1e = None
        self.s1e = None
        self.vhf = None
        self.fock = None

    def _common_init_(self, mcc):
        log = logger.Logger(mcc.stdout, mcc.verbose)

        self.mol = mcc.mol
        mf = mcc._scf
        # In PBC calculations, calculating h1e and vhf may be the computational
        # bottleneck. Thus, we allow using user precomputed results.
        dm = mf.make_rdm1()
        h1e = getattr(mcc, '_h1e', None)
        s1e = getattr(mcc, '_s1e', None)
        vhf = getattr(mcc, '_vhf', None)
        if h1e is None: h1e = mf.get_hcore()
        if s1e is None: s1e = mf.get_ovlp()
        if vhf is None:
            # vhf is used by CCSD solver and should be computed with exxdiv=None.
            if hasattr(mf, 'exxdiv'):
                log.warn('PBC HF is detected. vhf is computed with exxdiv=None.')
                with lib.temporary_env(mf, exxdiv=None):
                    vhf = mf.get_veff(dm=dm)
            else:
                vhf = mf.get_veff(dm=dm)
        elif hasattr(mf, 'exxdiv'):
            log.warn('Input vhf is detected while using PBC HF. Make sure '
                     'that the input vhf was computed with exxdiv=None, or '
                     'the MP2 and CCSD energy can be both wrong when compared '
                     'to k-point MP2 and CCSD results.')
        # fock is used to construct NOs. So we exxdiv-correct it if needed.
        if hasattr(mf, 'exxdiv') and mf.exxdiv == 'ewald':
            log.warn('PBC HF with exxdiv="ewald" is detected. fock is built '
                     'with exxdiv="ewald".')
            vhf_fock = mf.get_veff(dm=dm)
        else:
            vhf_fock = vhf
        fock = mf.get_fock(vhf=vhf_fock, dm=dm, h1e=h1e, s1e=s1e)
        vhf_fock = None

        self.h1e = h1e
        self.s1e = s1e
        self.vhf = vhf
        self.fock = fock

        return self

class _LNODFINCOREERIS(_LNOERIS):

    def _common_init_(self, mcc):
        log = logger.new_logger(mcc)
        _LNOERIS._common_init_(self, mcc)
        orbo, orbv = mcc.split_mo()[1:3]
        self.Lov = get_Lov(mcc._scf, orbo, orbv, log=log)

    def get_Ov(self, u):
        return einsum('iI,Lia->LIa', u, self.Lov)

    def get_oV(self, u):
        return einsum('aA,Lia->LiA', u, self.Lov)

    @staticmethod
    def _get_eris(Lia, Ljb):
        return einsum('Lia,Ljb->iajb', Lia, Ljb)

    def get_OvOv(self, u):
        LOv = self.get_Ov(self, u)
        return self._get_eris(LOv, LOv)

    def get_Ovov(self, u):
        LOv = self.get_Ov(u)
        return self._get_eris(LOv, self.Lov)

    def get_OvOv(self, u):
        LOv = self.get_Ov(u)
        return self._get_eris(LOv, LOv)

    def get_oVov(self, u):
        LoV = self.get_oV(u)
        return self._get_eris(LoV, self.Lov)

    def get_oVoV(self, u):
        LoV = self.get_oV(u)
        return self._get_eris(LoV, LoV)

def _make_df_eris_incore(mcc):
    eris = _LNODFINCOREERIS()
    eris._common_init_(mcc)
    return eris

def get_Lov(mf, orbo, orbv, Lov=None, log=None):
    import h5py
    from pyscf.ao2mo._ao2mo import nr_e2

    if log is None: log = logger.new_logger(mf)
    if not hasattr(mf, 'with_df'):
        raise RuntimeError

    cput0 = (logger.process_clock(), logger.perf_counter())

    nocc = orbo.shape[1]
    nvir = orbv.shape[1]
    nmo = nocc + nvir
    shls_slice = (0,nocc,nocc,nmo)
    naux = mf.with_df.get_naoaux()

    mo_coeff = np.asarray(np.hstack([orbo,orbv]), order='F')

    dtype = orbo.dtype
    dsize = orbo.itemsize
    nao = orbo.shape[0]
    nao_pair = nao*(nao+1)//2

    if Lov is None:
        mem_need = naux*nocc*nvir*8/1e6
        mem_avail = mf.max_memory - lib.current_memory()[0]
        log.debug1('ao2mo Lov mem_need= %.1f  mem_avail= %.1f', mem_need, mem_avail)
        if mem_avail < mem_need:
            log.error('Not enough memory for incore Lov (need %.1f MB, '
                      'available %.1f MB)', mem_avail, mem_need)
        Lov = np.empty([naux,nocc,nvir], dtype=dtype)

    cderi = mf.with_df._cderi
    if isinstance(cderi, np.ndarray):
        log.debug1('ao2mo found incore cderi  shape= %s', cderi.shape)
        feri = None
        get_Lpq_block = lambda p0,p1: cderi[p0:p1]
    else:
        feri = h5py.File(cderi, 'r')
        if getattr(mf.mol, 'pbc_intor', None):
            f3c = feri['j3c/0'] # 0 for Gamma point
        else:
            f3c = feri['j3c']
        log.debug1('ao2mo found outcore cderi  shape= %s', f3c['0'].shape)
        idxs = sorted([int(i) for i in list(f3c)])
        get_Lpq_block = lambda p0,p1: np.hstack([f3c['%d'%i][p0:p1]
                                                 for i in idxs])

    mem_avail = mf.max_memory - lib.current_memory()[0]
    blksize = int(np.floor(mem_avail*0.4 / ((nocc*nvir+nao_pair)*dsize/1e6)))
    log.debug1('ao2mo mem_avail= %.1f  blksize= %d  nblk= %d', mem_avail, blksize,
               naux//blksize+(1 if naux%blksize else 0))
    buf = np.empty([blksize,nocc*nvir], dtype=dtype)
    for p0,p1 in lib.prange(0,naux,blksize):
        log.debug1('ao2mo [%d:%d]', p0,p1)
        Lpq = get_Lpq_block(p0,p1)
        log.debug1('Lpq.shape= %s', Lpq.shape)
        nr_e2(Lpq, mo_coeff, shls_slice, aosym='s2', out=buf)
        Lpq = None
        Lov[p0:p1] = buf[:p1-p0].reshape(p1-p0,nocc,nvir)

    get_Lpq_block = Lpq = buf = None
    if feri is not None:
        f3c = None
        feri.close()

    log.timer("transforming DF-MP2 integrals", *cput0)

    return Lov


class _LNODFOUTCOREERIS(_LNODFINCOREERIS):

    def _common_init_(self, mcc):
        log = logger.new_logger(mcc)
        _LNOERIS._common_init_(self, mcc)
        orbo, orbv = mcc.split_mo()[1:3]
        naux = mcc._scf.with_df.get_naoaux()
        nocc = orbo.shape[1]
        nvir = orbv.shape[1]
        self.feri = lib.H5TmpFile()
        log.info('Lov is saved to %s', self.feri.filename)
        self.Lov = self.feri.create_dataset('Lov', (naux,nocc,nvir))
        get_Lov(mcc._scf, orbo, orbv, Lov=self.Lov, log=log)


def _make_df_eris_outcore(mcc):
    eris = _LNODFOUTCOREERIS()
    eris._common_init_(mcc)
    return eris

def _make_eris_incore(mcc):
    raise NotImplementedError

def _make_eris_outcore(mcc):
    raise NotImplementedError
