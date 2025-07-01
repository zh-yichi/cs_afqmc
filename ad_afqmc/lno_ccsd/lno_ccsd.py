import numpy as np
from jax import lax, vmap, jvp, random, jit
import jax
import jax.numpy as jnp
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf.lib import logger
from pyscf import __config__, ao2mo, mcscf, scf, lib
from ad_afqmc import pyscf_interface, wavefunctions
from ad_afqmc.lno.cc import ccsd
from functools import reduce
from ad_afqmc.wavefunctions import wave_function
from typing import Any, Tuple
import pickle
import h5py
from ad_afqmc import config

_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
#from ad_afqmc.lno.base import lno

def h1_frg(mf,orbfragloc):
    '''
        orbfragloc is the localized active occ that a set
        of active occ and vir orbitals are chosen for
    '''
    h1e = mf.get_hcore()
    nao = orbfragloc.shape[0]
    dm_i = np.einsum('j,k->jk',orbfragloc.reshape(nao),orbfragloc.reshape(nao))
    h1_i = np.einsum('ij,ji->',h1e,dm_i)*2
    return h1_i

def jk_frg(mf,orbfragloc,orbfrag,frzfrag):
    "Coulomb and exchange repulsion of fragment i"
    
    mol = mf.mol
    nao = mol.nao
    eri = mol.intor('int2e')
    dm_i = np.einsum('j,k->jk',orbfragloc.reshape(nao),orbfragloc.reshape(nao))
    act = np.array([i for i in range(mol.nao) if i not in frzfrag])
    nocc = mol.nelectron // 2
    act_occ = np.array([i for i in act if i < nocc])
    dm_act_occ = orbfrag[:, act_occ] @ orbfrag[:, act_occ].T
    en_j_frag = np.einsum("ij,kl,ijkl->",dm_i,dm_act_occ,eri)
    en_k_frag = np.einsum("il,kj,ijkl->",dm_i,dm_act_occ,eri)

    return 2*en_j_frag-en_k_frag

def e_elec_frg(mf,mo_frg,mo,frozen=[]):
    '''
    Fragemental electronic part of Hartree-Fock energy
    for given core hamiltonian and HF potential
    '''
    mol = mf.mol
    nao = mol.nao
    mo_frg = mo_frg.reshape(nao)
    dm_frg = 2*np.einsum('i,j->ij',mo_frg.T,mo_frg)
    nocc = np.count_nonzero(mf.mo_occ)
    act = np.array([i for i in range(nao) if i not in frozen])
    actocc = np.array([i for i in act if i < nocc])
    mo_acc = mo[:,actocc]
    dm_act = 2*np.einsum('nj,lj->nl',mo_acc,mo_acc)
    h1e = mf.get_hcore()
    vhf = mf.get_veff(mf.mol, dm_act)
    e1_frg = np.einsum('ij,ji->', h1e, dm_frg).real
    e_coul_frg = np.einsum('ij,ji->', vhf, dm_frg).real * .5
    return e1_frg+e_coul_frg

def _calc_olp_ratio_restricted(walker: jax.Array, wave_data: dict) -> complex:
    '''
    <psi_hf|walker>/<psi_ccsd|walker>
    '''
    nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
    GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    #o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
    o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
    o2 = 2 * jnp.einsum(
        "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]
    ) - jnp.einsum("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:])
    return 1/(1.0 + 2 * o1 + o2)

def cal_olp_ratio(walkers: jax.Array, wave_data: dict, trial) -> jax.Array:
    n_batch = trial.n_batch
    norb = trial.norb
    n_walkers = walkers.shape[0]
    batch_size = n_walkers // n_batch

    def scanned_fun(carry, walker_batch):
        overlap_batch = vmap(_calc_olp_ratio_restricted, in_axes=(0, None))(
            walker_batch, wave_data
        )
        return carry, overlap_batch

    _, overlaps = lax.scan(
        scanned_fun, None, walkers.reshape(n_batch, batch_size, norb, -1)
    )
    return overlaps.reshape(n_walkers)

def frg_hf_orb_cr(walkers,ham_data,wave_data,trial):
    hf_orb_cr = vmap(_frg_hf_orb_cr, in_axes=(None, None, 0, None, None))(
        ham_data['rot_h1'], ham_data['rot_chol'], walkers, trial, wave_data)
    return hf_orb_cr

def _frg_hf_orb_cr(rot_h1, rot_chol, walker, trial, wave_data):
    '''hf orbital energy multiplies the overlap ratio'''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    nocc = rot_h1.shape[0]
    _calc_green = wavefunctions.rhf(
        trial.norb,trial.nelec,n_batch=trial.n_batch)._calc_green
    green_walker = _calc_green(walker, wave_data)
    f = jnp.einsum('gij,jk->gik', rot_chol[:,:nocc,nocc:],
                    green_walker.T[nocc:,:nocc], optimize='optimal')
    c = vmap(jnp.trace)(f)
    eneo2Jt = jnp.einsum('Gxk,xk,G->',f,m,c)*2 
    eneo2ext = jnp.einsum('Gxy,Gyk,xk->',f,f,m)
    hf_orb_en = eneo2Jt - eneo2ext
    olp_ratio = _calc_olp_ratio_restricted(walker,wave_data)
    hf_orb_cr = jnp.real(olp_ratio*hf_orb_en)
    return hf_orb_cr

def _frg_modified_ccsd_olp_restricted(walker: jax.Array, wave_data: dict) -> complex:
    '''
    <psi_ccsd|walker>=<psi_0|walker>+C_ia^*G_ia+C_iajb^*(G_iaG_jb-G_ibG_ja)
    modified CCSD overlap returns the second and the third term
    that is, the overlap of the walker with the CCSD wavefunction
    without the hartree-fock part
    and skip one sum over the occ
    '''
    prjlo = wave_data["prjlo"].reshape(walker.shape[1])
    pick_i = jnp.where(abs(prjlo) > 1e-6, 1, 0)
    nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
    gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
    o1 = jnp.einsum("ia,ja->i", ci1, gf[:, nocc:])
    o2 = 2 * jnp.einsum("iajb,ia,jb->i", ci2, gf[:, nocc:], gf[:, nocc:]) \
        - jnp.einsum("iajb,ib,ja->i", ci2, gf[:, nocc:], gf[:, nocc:])
    olp = (2*o1+o2)*o0
    olp_i = jnp.einsum("i,i->", olp, pick_i)
    return olp_i

def frg_modified_ccsd_olp_restricted(walkers: jax.Array, wave_data: dict, trial) -> jax.Array:
    n_batch = trial.n_batch
    norb = trial.norb
    n_walkers = walkers.shape[0]
    #nocc = walkers.shape[2]
    batch_size = n_walkers // n_batch

    def scanned_fun(carry, walker_batch):
        overlap_batch = vmap(_frg_modified_ccsd_olp_restricted, in_axes=(0, None))(
            walker_batch, wave_data
        )
        return carry, overlap_batch

    _, overlaps = lax.scan(
        scanned_fun, None, walkers.reshape(n_batch, batch_size, norb, -1)
    )
    return overlaps.reshape(n_walkers)

## h0-E0 term
def _frg_elec_cr(walker,trial,wave_data,h0_E0):
    '''
    (h0-E0)*(1-N0(phi)/N_ccsd(phi))
    fragmental electron orbital energy correction
    '''
    mod_olp = _frg_modified_ccsd_olp_restricted(walker,wave_data)
    ccsd_olp = trial._calc_overlap_restricted(walker, wave_data)
    elec_cr = jnp.real(h0_E0*mod_olp/ccsd_olp)
    return elec_cr

def frg_elec_cr(walkers,trial,wave_data,h0_E0) -> jax.Array:

    elec_cr = vmap(_frg_elec_cr, in_axes=(0, None, None, None))(
        walkers, trial, wave_data, h0_E0)

    return elec_cr


def calc_hf_orbenergy(walkers,ham_data:dict,wave_data:dict,trial:wavefunctions) -> jnp.ndarray:
    return vmap(_calc_hf_orbenergy, in_axes=(None, None, 0, None, None))(
        ham_data['rot_h1'], ham_data['rot_chol'], walkers, trial, wave_data)

def _calc_hf_orbenergy(rot_h1, rot_chol, walker, trial, wave_data):
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    nocc = rot_h1.shape[0]
    _calc_green = wavefunctions.rhf(trial.norb,trial.nelec,n_batch=trial.n_batch)._calc_green
    green_walker = _calc_green(walker, wave_data)
    f = jnp.einsum('gij,jk->gik', rot_chol[:,:nocc,nocc:], green_walker.T[nocc:,:nocc], optimize='optimal')
    c = vmap(jnp.trace)(f)

    eneo2Jt = jnp.einsum('Gxk,xk,G->',f,m,c)*2 
    eneo2ext = jnp.einsum('Gxy,Gyk,xk->',f,f,m)
    return eneo2Jt - eneo2ext

def _thouless_linear(t,walker):
    new_walker = walker + t.dot(walker)
    return new_walker

def thouless_linear(t,walkers):
    new_walkers = vmap(_thouless_linear,in_axes=(None,0))(t,walkers)
    return new_walkers

def _frg_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    <psi_ccsd|exp(x*h1_mod)|walker>/<psi_ccsd|walker> without the hf part
    '''
    walker_1x = _thouless_linear(x*h1_mod, walker)
    olp = _frg_modified_ccsd_olp_restricted(walker_1x, wave_data)
    return olp

def _frg_olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    <psi_ccsd|exp(x*h2_mod)|walker>/<psi_ccsd|walker> without the hf part
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _frg_modified_ccsd_olp_restricted(walker_2x, wave_data)
    return olp

def _frg_ccsd_orb_cr(
    walker: jax.Array,
    ham_data: dict,
    wave_data: dict,
    trial: wavefunctions,
    eps :float = 1e-5
):
    '''
    one and two-body energy of a walker with ccsd trial wavefunction
    without the hf part
    '''

    norb = trial.norb
    chol = ham_data["chol"].reshape(-1, norb, norb)
    h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
    # v1 the one-body energy from the reordering of the 
    # two-body operators into non-normal ordered form
    v0 = 0.5 * jnp.einsum("gik,gjk->ij",
                            chol.reshape(-1, norb, norb),
                            chol.reshape(-1, norb, norb),
                            optimize="optimal")
    h1_mod = h1 - v0 
    ccsd_olp = trial._calc_overlap_restricted(walker, wave_data)

    x = 0.0
    # one body
    f1 = lambda a: _frg_olp_exp1(a,h1_mod,walker,wave_data)
    _, d_overlap = jvp(f1, [x], [1.0])

    # two body
    def scanned_fun(carry, c):
        eps, walker, wave_data = carry
        return carry, _frg_olp_exp2(eps,c,walker,wave_data)

    _, overlap_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, overlap_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, overlap_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

    ccsd_cr = jnp.real((d_overlap + jnp.sum(d_2_overlap) / 2.0) / ccsd_olp)
    return ccsd_cr

def frg_ccsd_orb_cr(
        walkers: jax.Array,
        ham_data: dict,
        wave_data: dict,
        trial: wavefunctions,
        eps :float = 1e-5) -> jax.Array:
    
    n_walkers = walkers.shape[0]
    # nocc = walkers.shape[2]
    batch_size = n_walkers // trial.n_batch

    def scanned_fun(carry, walker_batch):
        energy_batch = vmap(_frg_ccsd_orb_cr, in_axes=(0, None, None, None, None))(
            walker_batch, ham_data, wave_data, trial, eps
        )
        return carry, energy_batch

    _, energies = lax.scan(
        scanned_fun,
        None,
        walkers.reshape(trial.n_batch, batch_size, trial.norb, -1),
    )
    return energies.reshape(n_walkers)

def cc_impurity_solve(mf, mo_coeff, lo_coeff, ccsd_t=False, eris=None, frozen=None,
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
    mcc.ao2mo = ccsd.ccsd_ao2mo.__get__(mcc, mcc.__class__)
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
    elcorr_pt2 = ccsd.get_fragment_energy(oovv, t2, prjlo)
  
    cput1 = log.timer_debug1('imp sol - mp2 ene', *cput1)
    # CCSD fragment energy
    t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
    cput1 = log.timer_debug1('imp sol - cc  amp', *cput1)
    t2 += ccsd.einsum('ia,jb->ijab',t1,t1)
    elcorr_cc = ccsd.get_fragment_energy(oovv, t2, prjlo)
    cput1 = log.timer_debug1('imp sol - cc  ene', *cput1)
    if ccsd_t:
        from ad_afqmc.lno.cc.ccsd_t import kernel as CCSD_T
        t2 -= ccsd.einsum('ia,jb->ijab',t1,t1)   # restore t2
        elcorr_cc_t = CCSD_T(mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
        cput1 = log.timer_debug1('imp sol - cc  (T)', *cput1)
    else:
        elcorr_cc_t = 0.

    # frag_msg = '  '.join([f'E_corr(MP2) = {elcorr_pt2:.15g}',
    #                       f'E_corr(CCSD) = {elcorr_cc:.15g}',
    #                       f'E_corr(CCSD(T)) = {elcorr_cc_t:.15g}'])

    oovv = imp_eris = mcc = None

    return elcorr_cc, elcorr_cc_t, t1, t2

def write_dqmc(E0,hcore,hcore_mod,chol,nelec,nmo,enuc,ms=0,
    filename="FCIDUMP_chol",mo_coeffs=None):
    assert len(chol.shape) == 2
    with h5py.File(filename, "w") as fh5:
        fh5["header"] = np.array([nelec, nmo, ms, chol.shape[0]])
        fh5["hcore"] = hcore.flatten()
        fh5["hcore_mod"] = hcore_mod.flatten()
        fh5["chol"] = chol.flatten()
        fh5["energy_core"] = enuc
        fh5["E0"] = E0
        if mo_coeffs is not None:
            fh5["mo_coeffs_up"] = mo_coeffs[0]
            fh5["mo_coeffs_dn"] = mo_coeffs[1]

def prep_lno_amp_chol_file(mf_cc,mo_coeff,options,norb_act,nelec_act,
                           prjlo=[],norb_frozen=[],t1=None,t2=None,
                           chol_cut=1e-5,
                           option_file='options.bin',
                           mo_file="mo_coeff.npz",
                           amp_file="amplitudes.npz",
                           chol_file="FCIDUMP_chol"):
    
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    # write ccsd amplitudes
    if isinstance(mf_cc, (CCSD, UCCSD)):
        mf = mf_cc._scf
        #cc = mf_cc
    else:
        mf = mf_cc

    #     if isinstance(cc, UCCSD):
    if options["trial"] == "ucisd":
        ci2aa = t2[0] + 2 * np.einsum("ia,jb->ijab", t1[0], t1[0])
        ci2aa = (ci2aa - ci2aa.transpose(0, 1, 3, 2)) / 2
        ci2aa = ci2aa.transpose(0, 2, 1, 3)
        ci2bb = t2[2] + 2 * np.einsum("ia,jb->ijab", t1[1], t1[1])
        ci2bb = (ci2bb - ci2bb.transpose(0, 1, 3, 2)) / 2
        ci2bb = ci2bb.transpose(0, 2, 1, 3)
        ci2ab = t2[1] + np.einsum("ia,jb->ijab", t1[0], t1[1])
        ci2ab = ci2ab.transpose(0, 2, 1, 3)
        ci1a = np.array(t1[0])
        ci1b = np.array(t1[1])
        np.savez(
            amp_file,
            ci1a=ci1a,
            ci1b=ci1b,
            ci2aa=ci2aa,
            ci2ab=ci2ab,
            ci2bb=ci2bb,
        )
        
    elif options["trial"] == "cisd":
        ci2 = t2 + np.einsum("ia,jb->ijab", np.array(t1), np.array(t1))
        ci2 = ci2.transpose(0, 2, 1, 3)
        ci1 = np.array(t1)
        np.savez(amp_file, ci1=ci1, ci2=ci2)

    mol = mf.mol
    # calculate cholesky vectors
    h1e, chol, nelec, enuc, nbasis, nchol = [ None ] * 6
    print('# Generating Cholesky Integrals')

    mc = mcscf.CASSCF(mf, norb_act, nelec_act) 
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()

    nbasis = h1e.shape[-1]
    print(f'# frozen orbitals are {norb_frozen}')
    if isinstance(norb_frozen, (int, float)) and norb_frozen == 0:
        norb_frozen = []
    act = np.array([i for i in range(mol.nao) if i not in norb_frozen])
    print(f'# local active orbitals are {act}') #yichi
    print(f'# local active space size {len(act)}') #yichi
    e = ao2mo.kernel(mf.mol,mo_coeff[:,act],compact=False) # in mo representation
    print(f'# loc_eris shape: {e.shape}') #yichi
    # add e = pyscf_interface.df(mol_mf,e) for selected loc_mos
    chol = pyscf_interface.modified_cholesky(e,max_error = chol_cut) # in mo representation
    print(f'# chol shape: {chol.shape}') #yichi

    print("# Finished calculating Cholesky integrals\n")
    print('# Size of the correlation space:')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {nbasis}')
    print(f'# Number of Cholesky vectors: {chol.shape[0]}\n')
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        if isinstance(mf, scf.uhf.UHF):
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[0][:, norb_frozen:]))
            uhfCoeffs[:, :nbasis] = q
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[1][:, norb_frozen:]))
            uhfCoeffs[:, nbasis:] = q
        else:
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
            uhfCoeffs[:, :nbasis] = q
            uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        np.savez(mo_file,mo_coeff=trial_coeffs,prjlo=prjlo)

    elif isinstance(mf, scf.rhf.RHF):
        #q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
        q = np.eye(mol.nao- len(norb_frozen)) # atomic orbital
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez(mo_file,mo_coeff=trial_coeffs,prjlo=prjlo)

    write_dqmc(mf.e_tot,h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,
               filename=chol_file,mo_coeffs=trial_coeffs)
    
    return None


config.setup_jax()
MPI = config.setup_comm()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from functools import partial
from ad_afqmc import hamiltonian, propagation, sampling, run_afqmc

print = partial(print, flush=True)

# mo_file = run_afqmc.mo_file
# amp_file = run_afqmc.amp_file
# chol_file = run_afqmc.chol_file

def prep_lnoccsd_afqmc(options=None,prjlo=True,
                       option_file="options.bin",
                       mo_file="mo_coeff.npz",
                       amp_file="amplitudes.npz",
                       chol_file="FCIDUMP_chol"):
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    with h5py.File(chol_file, "r") as fh5:
        [nelec, nmo, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        e0 = jnp.array(fh5.get("E0"))
        h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
        chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    ms, nelec, nmo, nchol = int(ms), int(nelec), int(nmo), int(nchol)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)

    norb = nmo

    if options is None:
        try:
            with open(option_file, "rb") as f:
                options = pickle.load(f)
        except:
            options = {}

    options["dt"] = options.get("dt", 0.01)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse", "2rdm"]
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", True)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", "cisd")
    options["ene0"] = options.get("ene0", 0.0)
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    #options["orbE"] = options.get("orbE",0)
    options['maxError'] = options.get('maxError',1e-3)
    options["use_gpu"] = options.get("use_gpu", False)

    try:
        with h5py.File("observable.h5", "r") as fh5:
            [observable_constant] = fh5["constant"]
            observable_op = np.array(fh5.get("op")).reshape(nmo, nmo)
            if options["walker_type"] == "uhf":
                observable_op = jnp.array([observable_op, observable_op])
            observable = [observable_op, observable_constant]
    except:
        observable = None

    ham = hamiltonian.hamiltonian(nmo)
    ham_data = {}
    ham_data["h0"] = h0
    ham_data["E0"] = e0
    ham_data["h1"] = jnp.array([h1, h1])
    ham_data["chol"] = chol.reshape(nchol, -1)
    ham_data["ene0"] = options["ene0"]

    wave_data = {}
    if prjlo is True:
        prjlo = jnp.array(np.load(mo_file)["prjlo"])
    else: prjlo = None
    wave_data["prjlo"] = prjlo
    mo_coeff = jnp.array(np.load(mo_file)["mo_coeff"])
    wave_data["rdm1"] = jnp.array(
        [
            mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
            mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
        ]
    )

    wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]
    
    amplitudes = np.load(amp_file)
    ci1 = jnp.array(amplitudes["ci1"])
    ci2 = jnp.array(amplitudes["ci2"])
    trial_wave_data = {"ci1": ci1, "ci2": ci2}
    wave_data.update(trial_wave_data)
    trial = wavefunctions.cisd(norb, nelec_sp, n_batch=options["n_batch"])
    
    if options["walker_type"] == "rhf":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        #print(f'using {options["n_exp_terms"]} exp_terms')
        prop = propagation.propagator_restricted(
            options["dt"], 
            options["n_walkers"], 
            options["n_exp_terms"],
            options["n_batch"]
        )

    elif options["walker_type"] == "uhf":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        if options["free_projection"]:
            prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                10,
                n_batch=options["n_batch"],
            )
        else:
            prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                n_batch=options["n_batch"],
            )

    sampler = sampling.sampler(
        options["n_prop_steps"],
        options["n_ene_blocks"],
        options["n_sr_blocks"],
        options["n_blocks"],
    )

    if rank == 0:
        print(f"# norb: {norb}")
        print(f"# nelec: {nelec_sp}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI

### the following part of the code ###
###   is to TEST the convergence   ###
###  run on the full active space  ###
def _modified_ccsd_olp_restricted(walker: jax.Array, wave_data: dict) -> complex:
    '''
    <psi_ccsd|walker>=<psi_0|walker>+C_ia^*G_ia+C_iajb^*(G_iaG_jb-G_ibG_ja)
    modified CCSD overlap returns the second and the third term
    that is, the overlap of the walker with the CCSD wavefunction
    without the hartree-fock part
    '''
    nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
    gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
    o1 = jnp.einsum("ia,ja->", ci1, gf[:, nocc:])
    o2 = 2 * jnp.einsum("iajb,ia,jb->", ci2, gf[:, nocc:], gf[:, nocc:]) \
        - jnp.einsum("iajb,ib,ja->", ci2, gf[:, nocc:], gf[:, nocc:])
    olp = (2*o1+o2)*o0
    return olp

def modified_ccsd_olp_restricted(walkers: jax.Array, wave_data: dict, trial) -> jax.Array:
    n_batch = trial.n_batch
    norb = trial.norb
    n_walkers = walkers.shape[0]
    batch_size = n_walkers // n_batch

    def scanned_fun(carry, walker_batch):
        overlap_batch = vmap(_modified_ccsd_olp_restricted, in_axes=(0, None))(
            walker_batch, wave_data
        )
        return carry, overlap_batch
    
    _, overlaps = lax.scan(
        scanned_fun, None, walkers.reshape(n_batch, batch_size, norb, -1)
    )
    return overlaps.reshape(n_walkers)

def _olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    <psi_ccsd|exp(x*h1_mod)|walker>/<psi_ccsd|walker> without the hf part
    '''
    walker_1x = (
            walker
            + x * h1_mod.dot(walker)
        )
    olp = _modified_ccsd_olp_restricted(walker_1x, wave_data)
    return olp

def _olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    <psi_ccsd|exp(x*h2_mod)|walker>/<psi_ccsd|walker> without the hf part
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _modified_ccsd_olp_restricted(walker_2x, wave_data)
    return olp

def _tot_ccsd_cr(
    walker: jax.Array,
    ham_data: dict,
    wave_data: dict,
    trial: wavefunctions,
    eps :float = 1e-5
):
    """Calculates local energy using AD and finite difference for the two body term"""

    norb = trial.norb
    chol = ham_data["chol"].reshape(-1, norb, norb)
    h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
    # v1 the one-body energy from the reordering of the 
    # two-body operators into non-normal ordered form
    v0 = 0.5 * jnp.einsum("gik,gjk->ij",chol,chol,optimize="optimal")
    h1_mod = h1 - v0
    ccsd_olp = trial._calc_overlap_restricted(walker, wave_data)

    # one body
    x = 0.0
    f1 = lambda a: _olp_exp1(a,h1_mod,walker,wave_data)
    _, d_overlap = jvp(f1, [x], [1.0])

    # two body
    # carry: [eps, walker, wave_data]
    def scanned_fun(carry, x):
        eps, walker, wave_data = carry
        return carry, _olp_exp2(eps,x,walker,wave_data)

    _, overlap_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, overlap_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, overlap_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

    ccsd_cr = jnp.real((d_overlap + jnp.sum(d_2_overlap) / 2.0) / ccsd_olp)
    return ccsd_cr

def tot_ccsd_cr(
        walkers: jax.Array, 
        ham_data: dict, 
        wave_data: dict, 
        trial: wavefunctions,
        eps = 1e-5) -> jax.Array:
    n_walkers = walkers.shape[0]
    batch_size = n_walkers // trial.n_batch

    def scanned_fun(carry, walker_batch):
        energy_batch = vmap(_tot_ccsd_cr, in_axes=(0, None, None, None, None))(
            walker_batch, ham_data, wave_data, trial, eps
        )
        return carry, energy_batch

    _, energies = lax.scan(
        scanned_fun,
        None,
        walkers.reshape(trial.n_batch, batch_size, trial.norb, -1),
    )
    return energies.reshape(n_walkers)

# sampler_eq = sampling.sampler(n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10)
@partial(jit, static_argnums=(2,3,5))
def block_tot(prop_data: dict,
               ham_data: dict,
               prop: propagation.propagator,
               trial: wave_function,
               wave_data: dict,
               sampler: sampling.sampler):
        """Block scan function. Propagation and calculate total ccsd correction energy."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                sampler.n_prop_steps,
                prop.n_walkers,
                ham_data["chol"].shape[0],
            ),
        )
        _step_scan_wrapper = lambda x, y: sampler._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )
        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        ccsd_cr = tot_ccsd_cr(
            prop_data["walkers"],ham_data,wave_data,trial,1e-5)
        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )
        blk_wt = jnp.sum(prop_data["weights"])
        blk_ccsd_cr = jnp.sum(ccsd_cr * prop_data["weights"]) / blk_wt
        blk_energy = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_energy
        )
        
        return prop_data,(blk_ccsd_cr,blk_wt)

@partial(jit, static_argnums=(2,3,5))
def _sr_block_scan_tot(
    prop_data: dict,
    ham_data: dict,
    prop: propagation.propagator,
    trial: wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    def _block_scan_wrapper(x,_):
        return block_tot(x,ham_data,prop,trial,wave_data,sampler)
    
    # propagate n_ene_blocks then do sr
    prop_data, (blk_ccsd_tot_cr,blk_wt) \
        = lax.scan(
        _block_scan_wrapper, prop_data, xs=None, length=sampler.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    return prop_data, (blk_ccsd_tot_cr,blk_wt)

@partial(jit, static_argnums=(1, 3, 5))
def propagate_phaseless_tot(
    ham_data: dict,
    prop: propagation.propagator,
    prop_data: dict,
    trial: wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[jax.Array, dict]:
    def _sr_block_scan_wrapper(x,_):
        return _sr_block_scan_tot(x, ham_data, prop, trial, wave_data, sampler)

    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data,(blk_ccsd_tot_cr,blk_wt) \
        = lax.scan(
        _sr_block_scan_wrapper, prop_data, xs=None, length=sampler.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sampler.n_sr_blocks * sampler.n_ene_blocks * prop.n_walkers
    )
    wt = jnp.sum(blk_wt)
    ccsd_tot_cr = jnp.sum(blk_ccsd_tot_cr * blk_wt) / wt
    return prop_data, (ccsd_tot_cr,wt)
##################################################

# def one_block_orb(prop_data,ham_data,prop,trial,wave_data,hf_elec_frg):
#         """Block scan function. Propagation and orbital_i energy calculation."""
#         prop_data["key"], subkey = random.split(prop_data["key"])
#         fields = random.normal(
#             subkey,
#             shape=(
#                 sampler_eq.n_prop_steps,
#                 prop.n_walkers,
#                 ham_data["chol"].shape[0],
#             ),
#         )
#         _step_scan_wrapper = lambda x, y: sampler_eq._step_scan(
#             x, y, ham_data, prop, trial, wave_data
#         )
#         prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
#         prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
#             prop_data["weights"]
#         )
#         prop_data = prop.orthonormalize_walkers(prop_data)
#         prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
#         hf_orb_energy = calc_hf_orbenergy(prop_data["walkers"],ham_data,wave_data,trial)
#         olp_ratio = cal_olp_ratio(prop_data["walkers"], wave_data,trial)
#         ccsd_orb_energy = frg_energy_ccsd_restricted(prop_data["walkers"], ham_data, 
#                                         wave_data, trial,1e-5)
#         hf_correction = -(1-olp_ratio)*hf_elec_frg
#         orb_energy_samples = jnp.real(hf_correction + olp_ratio*hf_orb_energy + ccsd_orb_energy)
#         energy_samples = jnp.real(
#             trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
#         )
#         energy_samples = jnp.where(
#             jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
#             prop_data["e_estimate"],
#             energy_samples,
#         )
#         block_weight = jnp.sum(prop_data["weights"])
#         block_orb_energy = jnp.sum(orb_energy_samples * prop_data["weights"]) / block_weight
#         block_energy = jnp.sum(energy_samples * prop_data["weights"]) / block_weight
#         prop_data["pop_control_ene_shift"] = (
#             0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
#         )
#         return prop_data, (block_orb_energy, block_weight)

@partial(jit, static_argnums=(2,3,5))
def block_orb(prop_data: dict,
              ham_data: dict,
              prop: propagation.propagator,
              trial: wave_function,
              wave_data: dict,
              sampler: sampling.sampler):
        """Block scan function. Propagation and orbital_i energy calculation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                sampler.n_prop_steps,
                prop.n_walkers,
                ham_data["chol"].shape[0],
            ),
        )
        # propgate n_prop_steps x dt
        _step_scan_wrapper = lambda x, y: sampler._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )
        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        #olp_ratio = jnp.real(cal_olp_ratio(prop_data["walkers"], wave_data,trial))
        # h0-E0 term #
        h0_e0 = ham_data["h0"]-ham_data["E0"]
        elec_orb_cr = jnp.real(frg_elec_cr(prop_data["walkers"],trial,wave_data,h0_e0))

        hf_orb_cr = jnp.real(frg_hf_orb_cr(prop_data["walkers"],ham_data,wave_data,trial))
        ccsd_orb_cr = jnp.real(frg_ccsd_orb_cr(prop_data["walkers"], ham_data, 
                                        wave_data, trial,1e-5))
        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )
        blk_wt = jnp.sum(prop_data["weights"])
        blk_elec_orb_cr = jnp.sum(elec_orb_cr * prop_data["weights"]) / blk_wt
        blk_hf_orb_cr = jnp.sum(hf_orb_cr * prop_data["weights"]) / blk_wt
        blk_ccsd_orb_cr = jnp.sum(ccsd_orb_cr * prop_data["weights"]) / blk_wt
        #blk_olp_ratio = jnp.sum(olp_ratio * prop_data["weights"]) / blk_wt
        blk_energy = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_energy
        )
        blk_orb_cr = blk_elec_orb_cr+blk_hf_orb_cr+blk_ccsd_orb_cr

        return prop_data,(blk_energy,blk_elec_orb_cr,blk_hf_orb_cr,blk_ccsd_orb_cr,blk_orb_cr,blk_wt)

@partial(jit, static_argnums=(1, 3, 5))
def propagate_phaseless_orb(
    ham_data: dict,
    prop: propagation.propagator,
    prop_data: dict,
    trial: wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[jax.Array, dict]:
    def _sr_block_scan_wrapper(x,_):
        return _sr_block_scan_orb(x, ham_data, prop, trial, wave_data, sampler)

    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data,(blk_energy,blk_elec_orb_cr,blk_hf_orb_cr,blk_ccsd_orb_cr,blk_orb_cr,blk_wt) \
        = lax.scan(
        _sr_block_scan_wrapper, prop_data, xs=None, length=sampler.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sampler.n_sr_blocks * sampler.n_ene_blocks * prop.n_walkers
    )
    wt = jnp.sum(blk_wt)
    energy = jnp.sum(blk_energy * blk_wt) / wt
    elec_orb_cr = jnp.sum(blk_elec_orb_cr * blk_wt) / wt
    hf_orb_cr = jnp.sum(blk_hf_orb_cr * blk_wt) / wt
    ccsd_orb_cr = jnp.sum(blk_ccsd_orb_cr * blk_wt) / wt
    orb_cr = jnp.sum(blk_orb_cr * blk_wt) / wt

    return prop_data, (energy,elec_orb_cr,hf_orb_cr,ccsd_orb_cr,orb_cr,wt)

@partial(jit, static_argnums=(2,3,5))
def _sr_block_scan_orb(
    prop_data: dict,
    ham_data: dict,
    prop: propagation.propagator,
    trial: wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    def _block_scan_wrapper(x,_):
        return block_orb(x,ham_data,prop,trial,wave_data,sampler)
    
    # propagate n_ene_blocks then do sr
    prop_data, (blk_energy,blk_elec_orb_cr,blk_hf_orb_cr,blk_ccsd_orb_cr,blk_orb_cr,blk_wt) \
        = lax.scan(
        _block_scan_wrapper, prop_data, xs=None, length=sampler.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

    return prop_data, (blk_energy,blk_elec_orb_cr,blk_hf_orb_cr,blk_ccsd_orb_cr,blk_orb_cr,blk_wt)

import sys, os
from ad_afqmc.lno.cc import LNOCCSD
from ad_afqmc.lno.afqmc import LNOAFQMC
from ad_afqmc.lno.base import lno
from pyscf.lib import logger
log = logger.Logger(sys.stdout, 6)

def get_fragment_energy(oovv, t2, prj):
    m = fdot(prj.T, prj)
    return lib.einsum('ijab,kjab,ik->',t2,2*oovv-oovv.transpose(0,1,3,2),m)

def run_lno_ccsd_afqmc(mf,thresh,frozen,options,chol_cut,nproc,run_frg_list=None,mp2=False):
    '''
    mf: pyscf mean-field object
    thresh: lno thresh
    frozen: frozen orbitals
    options: afqmc options
    chol_cut: Cholesky Decomposition cutoff
    nproc: number of processors
    run_frg_list: list of the fragments to run
    '''

    lo_type = 'pm'
    no_type = 'ie' # cim
    frag_lolist = '1o'

    _fdot = np.dot
    fdot = lambda *args: reduce(_fdot, args)

    if isinstance(thresh, float):
        thresh_occ = thresh*10
        thresh_vir = thresh
    else:
        thresh_occ, thresh_vir = thresh

    lno_cc = LNOCCSD(mf, thresh=thresh, frozen=frozen)
    lno_cc.thresh_occ = thresh_occ
    lno_cc.thresh_vir = thresh_vir
    lno_cc.lo_type = lo_type
    lno_cc.no_type = no_type
    lno_cc.frag_lolist = frag_lolist
    lno_cc.ccsd_t = True
    lno_cc.force_outcore_ao2mo = True
    orbloc = lno_cc.orbloc
    # lo_type = lno_cc.lo_type
    # no_type = lno_cc.no_type
    frag_atmlist = lno_cc.frag_atmlist
    # frag_lolist = lno_cc.frag_lolist
    s1e = lno_cc._scf.get_ovlp()

    lno_qmc = LNOAFQMC(mf, thresh=thresh, frozen=frozen)
    lno_qmc.thresh_occ = thresh_occ
    lno_qmc.thresh_vir = thresh_vir
    lno_qmc.nblocks = options["n_blocks"]
    lno_qmc.nwalk_per_proc = options["n_walkers"]
    lno_qmc.nproc = nproc
    lno_qmc.lo_type = lo_type
    lno_qmc.no_type = no_type
    lno_qmc.frag_lolist = frag_lolist
    lno_qmc.chol_cut = chol_cut

    # NO type
    # no_type = 'ie'
    # frag_lolist = '1o'
    log.info('no_type = %s', no_type)

    # LO construction
    orbloc = lno_cc.get_lo(lo_type=lo_type) # localized active occ orbitals
    orbactocc = lno_cc.split_mo()[1] # non-localized active occ
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

    if frag_lolist == '1o':
        log.info('Using single-LO fragment') # this is what we use, every active local occ labels a fragment
        frag_lolist = [[i] for i in range(orbloc.shape[1])]
    else: print('Only support single LO fragment!')
    nfrag = len(frag_lolist)
    frag_nonvlist = lno_cc.frag_nonvlist

    # dump info
    log.info('nfrag = %d  nlo = %d', nfrag, orbloc.shape[1])
    log.info('frag_atmlist = %s', frag_atmlist)
    log.info('frag_lolist = %s', frag_lolist)
    log.info('frag_nonvlist = %s', frag_nonvlist)

    if not (no_type[0] in 'rei' and no_type[1] in 'rei'):
        log.warn('Input no_type "%s" is invalid.', no_type)
        raise ValueError

    if frag_nonvlist is None: frag_nonvlist = [[None,None]] * nfrag

    eris = lno_cc.ao2mo()
    frozen_mask = lno_cc.get_frozen_mask()
    thresh_pno = [thresh_occ,thresh_vir]
    print(f'lno thresh {thresh_pno}')
    
    if run_frg_list is None:
        run_frg_list = range(nfrag)

    for ifrag in run_frg_list:
        print(f'########### running fragment {ifrag+1} ##########')
        #if(len(lno_cc.runfrags)>0):
        #    if(ifrag not in lno_cc.runfrags):frag_res[ifrag] = (0,0,0)
        fraglo = frag_lolist[ifrag]
        orbfragloc = orbloc[:,fraglo] # the specific local active occ
        frag_target_nocc, frag_target_nvir = frag_nonvlist[ifrag]
        THRESH_INTERNAL = 1e-10
        frzfrag, orbfrag, can_orbfrag = lno.make_fpno1(lno_cc, eris, orbfragloc, no_type,
                                                    THRESH_INTERNAL, thresh_pno,
                                                    frozen_mask=frozen_mask,
                                                    frag_target_nocc=frag_target_nocc,
                                                    frag_target_nvir=frag_target_nvir,
                                                    canonicalize=False)

        ecorr_ccsd,_,t1,t2 = cc_impurity_solve(
            mf,orbfrag,orbfragloc,frozen=frzfrag,eris=eris,log=log)
        print(f'# lno-ccsd correlation energy is: {ecorr_ccsd}')
        #frozen=frzfrag

        maskocc = mf.mo_occ>1e-10
        nmo = mf.mo_occ.size

        # Convert frozen to 0 bc PySCF solvers do not support frozen=None or empty list
        if frzfrag is None:
            frzfrag = 0
        elif isinstance(frzfrag, (list,tuple,np.ndarray)) and len(frzfrag) == 0:
            frzfrag = 0

        if isinstance(frzfrag, (int,np.integer)):
            maskact = np.hstack([np.zeros(frzfrag,dtype=bool),
                                    np.ones(nmo-frzfrag,dtype=bool)])
        elif isinstance(frzfrag, (list,tuple,np.ndarray)):
            maskact = np.array([i not in frzfrag for i in range(nmo)])
        else:
            raise RuntimeError

        orbfrzocc = orbfrag[:,~maskact& maskocc]
        orbactocc = orbfrag[:, maskact& maskocc]
        orbactvir = orbfrag[:, maskact&~maskocc]
        orbfrzvir = orbfrag[:,~maskact&~maskocc]
        _, nactocc, nactvir, _ = [orb.shape[1]
                                                for orb in [orbfrzocc,orbactocc,
                                                            orbactvir,orbfrzvir]]
        s1e = mf.get_ovlp() if eris is None else eris.s1e
        prjlo = fdot(orbfragloc.T, s1e, orbactocc) ### overlap between the lo and each active occ in its fragment
        #print('# lo projection on its active occupied fragment subspace', prjlo)

        prep_lno_amp_chol_file(
            mf,orbfrag,options,
            norb_act=(nactocc+nactvir),nelec_act=nactocc*2,
            prjlo=prjlo,norb_frozen=frzfrag,
            t1=t1,t2=t2
                    )
        
        #MP2 correction 
        if mp2:
            log.info('running fragment MP2')
            from pyscf.cc import CCSD
            s1e = mf.get_ovlp() if eris is None else eris.s1e
            can_prjlo = fdot(orbfragloc.T, s1e, can_orbfrag[:, maskact& maskocc])
            mcc = CCSD(mf, mo_coeff=can_orbfrag, frozen=frzfrag)
            # print(f'######{can_orbfrag.shape,orbfrag.shape}######')
            mcc.ao2mo = ccsd.ccsd_ao2mo.__get__(mcc, mcc.__class__)
            mcc._s1e = s1e
            if eris is not None:
                mcc._h1e = eris.h1e
                mcc._vhf = eris.vhf
            imp_eris = mcc.ao2mo()

            if isinstance(imp_eris.ovov, np.ndarray):
                ovov = imp_eris.ovov
            else:
                ovov = imp_eris.ovov[()]
            oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
            ovov = None
            #
            # MP2 fragment energy
            _, t2 = mcc.init_amps(eris=imp_eris)[1:]
            elcorr_pt2 = get_fragment_energy(oovv, t2, can_prjlo)
            print(f'# lno-mp2 fragment energy: {elcorr_pt2:.6f}')

        # Run AFQMC
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/run_lnocc_frg.py"
        os.system(
            f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} |tee frg_{ifrag+1}.out"
        )

        with open(f'frg_{ifrag+1}.out', 'a') as out_file:
            if mp2:
                print(f"lno-mp2 orb_corr: {elcorr_pt2:.6f}",file=out_file)
            print(f"lno-ccsd orb_corr: {ecorr_ccsd:.6f}",file=out_file)

    with open('results.out', 'w') as out_file:
        print('# frag \t elec_orb_cr \t err \t hf_orb_cr \t err \t ccsd_orb_cr \t err \t e_afqmc_orb_cr \t err \t e_mp2_orb_corr \t e_ccsd_orb_corr',file=out_file)
        for ifrag in range(nfrag):
            with open(f"frg_{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "lno-ccsd-afqmc elec_orb_cr" in line:
                        elec_orb_cr = line.split()[2]
                        elec_orb_cr_err = line.split()[4]
                    if "lno-ccsd-afqmc hf_orb_cr" in line:
                        hf_orb_cr = line.split()[2]
                        hf_orb_cr_err = line.split()[4]
                    if "lno-ccsd-afqmc ccsd_orb_cr" in line:
                        ccsd_orb_cr = line.split()[2]
                        ccsd_orb_cr_err = line.split()[4]
                    if "lno-ccsd-afqmc tot_orb_cr" in line:
                        tot_orb_cr = line.split()[2]
                        tot_orb_cr_err = line.split()[4]
                    if mp2:
                        if "lno-mp2 orb_corr" in line:
                            e_mp2_orb_cr = line.split()[2]
                    else:
                        e_mp2_orb_cr = '  None  '
                    if "lno-ccsd orb_corr" in line:
                        e_ccsd_orb_corr = line.split()[2]
                print(f'{ifrag+1:3d} \t {elec_orb_cr} \t {elec_orb_cr_err} \t {hf_orb_cr} \t {hf_orb_cr_err} \t {ccsd_orb_cr} \t {ccsd_orb_cr_err} \t {tot_orb_cr} \t {tot_orb_cr_err} \t {e_mp2_orb_cr} \t {e_ccsd_orb_corr}', file=out_file)

    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == '  None  '] = '0.000000' 
                # elec_orb_cr = line.split()[1]
                # elec_orb_cr_err = line.split()[2]
                # print(elec_orb_cr, elec_orb_cr_err)
    data = np.array(data.reshape(3,11))
    elec_orb_cr = np.array(data[:,1],dtype='float32')
    elec_orb_cr_err = np.array(data[:,2],dtype='float32')
    hf_orb_cr = np.array(data[:,3],dtype='float32')
    hf_orb_cr_err = np.array(data[:,4],dtype='float32')
    ccsd_orb_cr = np.array(data[:,5],dtype='float32')
    ccsd_orb_cr_err = np.array(data[:,6],dtype='float32')
    tot_orb_cr = np.array(data[:,7],dtype='float32')
    tot_orb_cr_err = np.array(data[:,8],dtype='float32')
    e_mp2_orb_cr = np.array(data[:,9],dtype='float32')
    e_ccsd_orb_cr = np.array(data[:,10],dtype='float32')
    elec_cr = sum(elec_orb_cr)
    elec_cr_err = np.sqrt(sum(elec_orb_cr_err**2))
    hf_cr = sum(hf_orb_cr)
    hf_cr_err = np.sqrt(sum(hf_orb_cr_err**2))
    ccsd_cr = sum(ccsd_orb_cr)
    ccsd_cr_err = np.sqrt(sum(ccsd_orb_cr_err**2))
    tot_cr = sum(tot_orb_cr)
    tot_cr_err = np.sqrt(sum(tot_orb_cr_err**2))
    e_mp2_corr = sum(e_mp2_orb_cr)
    e_ccsd_corr = sum(e_ccsd_orb_cr)

    if mp2:
        from pyscf import mp
        mmp = mp.MP2(mf, frozen=frozen)
        e_mp2 = mmp.kernel()[0]
        mp2_corrected_tot_cr = tot_cr + e_mp2 - e_mp2_corr
        mp2_corrected_tot_cr = f'{mp2_corrected_tot_cr:.6f}'

    elec_cr = f'{elec_cr:.6f}'
    elec_cr_err = f'{elec_cr_err:.6f}'
    hf_cr = f'{hf_cr:.6f}'
    hf_cr_err = f'{hf_cr_err:.6f}'
    ccsd_cr = f'{ccsd_cr:.6f}'
    ccsd_cr_err = f'{ccsd_cr_err:.6f}'
    tot_cr = f'{tot_cr:.6f}'
    tot_cr_err = f'{tot_cr_err:.6f}'
    if mp2:
        e_mp2_corr = f'{e_mp2_corr:.6f}'
    else:
        e_mp2_corr = '  None  '
    e_ccsd_corr = f'{e_ccsd_corr:.6f}'

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# elec_cr {elec_cr} +/- {elec_cr_err}\n')
        out_file.write(f'# hf_cr {hf_cr} +/- {hf_cr_err}\n')
        out_file.write(f'# ccsd_cr {ccsd_cr} +/- {ccsd_cr_err}\n')
        out_file.write(f'# tot_afqmc_cr {tot_cr} +/- {tot_cr_err}\n')
        out_file.write(f'# e_mp2_corr {e_mp2_corr}\n')
        out_file.write(f'# e_ccsd_corr {e_ccsd_corr}\n')
        if mp2:
            out_file.write(f'# mp2_corrected tot_afqmc_cr {mp2_corrected_tot_cr}\n')
    return None