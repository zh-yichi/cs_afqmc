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
from typing import Tuple
import pickle
import h5py
from ad_afqmc import config
from functools import partial
from ad_afqmc import hamiltonian, propagation, sampling

print = partial(print, flush=True)

_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
#from ad_afqmc.lno.base import lno


@jax.jit
def _calc_olp_ratio_restricted(walker: jax.Array, wave_data: dict) -> complex:
    '''
    <psi_hf|walker>/<psi_ccsd|walker>
    '''
    nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
    GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    #o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
    o1 = jnp.einsum("ia,ia->", ci1, GF[:, nocc:])
    o2 = 2 * jnp.einsum("iajb, ia, jb->", ci2, GF[:, nocc:], GF[:, nocc:]) \
        - jnp.einsum("iajb, ib, ja->", ci2, GF[:, nocc:], GF[:, nocc:])
    return 1/(1.0 + 2 * o1 + o2)

@partial(jit, static_argnums=(2,))
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

@partial(jit, static_argnums=(3,))
def _frg_hf_cr(rot_h1, rot_chol, walker, trial, wave_data):
    '''hf orbital correlation energy multiplies the overlap ratio'''
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
    return hf_orb_cr, jnp.real(hf_orb_en)

@partial(jit, static_argnums=(3,))
def frg_hf_cr(walkers,ham_data,wave_data,trial):
    hf_orb_cr, hf_orb_en = vmap(_frg_hf_cr, in_axes=(None, None, 0, None, None))(
        ham_data['rot_h1'], ham_data['rot_chol'], walkers, trial, wave_data)
    return hf_orb_cr, hf_orb_en

@jax.jit
def _frg_modified_ccsd_olp_restricted(walker: jax.Array, wave_data: dict) -> complex:
    '''
    <psi_ccsd|walker>=<psi_0|walker>+C_ia^*G_ia+C_iajb^*(G_iaG_jb-G_ibG_ja)
    modified CCSD overlap returns the second and the third term
    that is, the overlap of the walker with the CCSD wavefunction
    without the hartree-fock part
    and skip one sum over the occ
    '''
    # prjlo = wave_data["prjlo"].reshape(walker.shape[1])
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
    gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    # pick_i = jnp.where(abs(prjlo) > 1e-6, 1, 0)
    # o1 = jnp.einsum("ia,ia->i", ci1, gf[:, nocc:])
    # o2 = 2 * jnp.einsum("iajb,ia,jb->i", ci2, gf[:, nocc:], gf[:, nocc:]) \
    #     - jnp.einsum("iajb,ib,ja->i", ci2, gf[:, nocc:], gf[:, nocc:])
    o1 = jnp.einsum("ia,ka,ik->", ci1, gf[:, nocc:],m)
    o2 = 2 * jnp.einsum("iajb,ka,jb,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m) \
        - jnp.einsum("iajb,kb,ja,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m)
    olp = (2*o1+o2)*o0
    # olp_i = jnp.einsum("i,i->", olp, pick_i)
    return olp

@jax.jit
def _frg_modified_ccsd_olp_restricted2(walker: jax.Array, wave_data: dict) -> complex:
    '''
    <psi_ccsd|walker>=<psi_0|walker>+C_ia^*G_ia+C_iajb^*(G_iaG_jb-G_ibG_ja)
    modified CCSD overlap returns the second and the third term
    that is, the overlap of the walker with the CCSD wavefunction
    without the hartree-fock part
    and skip one sum over the occ
    '''
    # prjlo = wave_data["prjlo"].reshape(walker.shape[1])
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    nocc, ci1, ci2 = walker.shape[1], wave_data["full_ci1"], wave_data["full_ci2"]
    gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    # pick_i = jnp.where(abs(prjlo) > 1e-6, 1, 0)
    # o1 = jnp.einsum("ia,ia->i", ci1, gf[:, nocc:])
    # o2 = 2 * jnp.einsum("iajb,ia,jb->i", ci2, gf[:, nocc:], gf[:, nocc:]) \
    #     - jnp.einsum("iajb,ib,ja->i", ci2, gf[:, nocc:], gf[:, nocc:])
    o1 = jnp.einsum("ia,ka,ik->", ci1, gf[:, nocc:],m)
    o2 = 2 * jnp.einsum("iajb,ka,jb,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m) \
        - jnp.einsum("iajb,kb,ja,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m)
    olp = (2*o1+o2)*o0
    # olp_i = jnp.einsum("i,i->", olp, pick_i)
    return olp

@partial(jit, static_argnums=(2,))
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

@partial(jit, static_argnums=(3,))
def calc_hf_orbenergy(walkers,ham_data:dict,wave_data:dict,trial:wavefunctions) -> jnp.ndarray:
    return vmap(_calc_hf_orbenergy, in_axes=(None, None, 0, None, None))(
        ham_data['rot_h1'], ham_data['rot_chol'], walkers, trial, wave_data)

@partial(jit, static_argnums=(3,))
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

@jax.jit
def _thouless_linear(t,walker):
    new_walker = walker + t.dot(walker)
    return new_walker

@jax.jit
def thouless_linear(t,walkers):
    new_walkers = vmap(_thouless_linear,in_axes=(None,0))(t,walkers)
    return new_walkers

@jax.jit
def _frg_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    <psi_ccsd|exp(x*h1_mod)|walker>/<psi_ccsd|walker> without the hf part
    '''
    walker_1x = _thouless_linear(x*h1_mod, walker)
    olp = _frg_modified_ccsd_olp_restricted(walker_1x, wave_data)
    return olp

@jax.jit
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

@partial(jit, static_argnums=(3,))
def _frg_ccsd_cr(
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
    
    # zero body
    h0_E0 = ham_data["h0"]-ham_data["E0"]
    mod_olp = _frg_modified_ccsd_olp_restricted(walker,wave_data)

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

    # ccsd_cr0 = jnp.real(h0_E0*mod_olp/ccsd_olp)
    # ccsd_cr1 = jnp.real(d_overlap/ccsd_olp)
    # ccsd_cr2 = jnp.real(0.5*jnp.sum(d_2_overlap)/ccsd_olp)

    ccsd_cr = jnp.real(
        (h0_E0*mod_olp + d_overlap + jnp.sum(d_2_overlap) / 2.0) / ccsd_olp)

    return ccsd_cr #ccsd_cr0,ccsd_cr1,ccsd_cr2,

@partial(jit, static_argnums=(3,))
def frg_ccsd_cr(
        walkers: jax.Array,
        ham_data: dict,
        wave_data: dict,
        trial: wavefunctions,
        eps :float = 1e-5) -> jax.Array:
    
    ccsd_cr = vmap(_frg_ccsd_cr, 
                   in_axes=(0, None, None, None, None))(
                       walkers, ham_data, wave_data, trial, eps)

    return ccsd_cr #ccsd_cr0,ccsd_cr1,ccsd_cr2

# calculate the overlap of a full cisd wavefunction (in MO) with a walker (in NO)
@jax.jit
def no2mo(mo_coeff,s1e,no_coeff):
    prj = mo_coeff.T@s1e@no_coeff
    return prj

@jax.jit
def prj_walker(p_frzocc,p_act,walker):
    walker_act = p_act@walker
    walker_new = jnp.hstack((p_frzocc,walker_act))
    return walker_new

@jax.jit
def cisd_walker_overlap_ratio(walker,ci1,ci2):
    nocc = walker.shape[1]
    GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    # o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
    o2 = 2 * jnp.einsum(
        "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]
    ) - jnp.einsum("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:])
    return 1/(1.0 + 2 * o1 + o2)

@jax.jit
def cisd_walker_olp(walker,ci1,ci2):
    nocc = walker.shape[1]
    GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
    o2 = 2 * jnp.einsum(
        "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]
    ) - jnp.einsum("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:])
    return (1.0 + 2 * o1 + o2) * o0

### debug mode ###
@partial(jit, static_argnums=(3,))
def _frg_hf_cr_dbg(rot_h1, rot_chol, walker, trial, wave_data):
    '''hf orbital correlation energy multiplies the overlap ratio'''
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
    return jnp.abs(olp_ratio), hf_orb_cr

@partial(jit, static_argnums=(3,))
def frg_hf_cr_dbg(walkers,ham_data,wave_data,trial):
    olp_r, hf_orb_cr = vmap(_frg_hf_cr_dbg, in_axes=(None, None, 0, None, None))(
        ham_data['rot_h1'], ham_data['rot_chol'], walkers, trial, wave_data)
    return olp_r, hf_orb_cr

@partial(jit, static_argnums=(3,))
def _frg_cc_cr_dbg(
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
    
    # zero body
    h0_E0 = ham_data["h0"]-ham_data["E0"]
    mod_olp = _frg_modified_ccsd_olp_restricted(walker,wave_data)

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

    cc_cr0 = jnp.real(h0_E0*mod_olp/ccsd_olp)
    cc_cr1 = jnp.real(d_overlap/ccsd_olp)
    cc_cr2 = jnp.real(0.5*jnp.sum(d_2_overlap)/ccsd_olp)
    cc_cr = cc_cr0 + cc_cr1 + cc_cr2

    return cc_cr0,cc_cr1,cc_cr2,cc_cr

@partial(jit, static_argnums=(3,))
def frg_cc_cr_dbg(
        walkers: jax.Array,
        ham_data: dict,
        wave_data: dict,
        trial: wavefunctions,
        eps :float = 1e-5) -> jax.Array:
    
    cc_cr0,cc_cr1,cc_cr2,cc_cr \
        = vmap(_frg_cc_cr_dbg, in_axes=(0, None, None, None, None))(
        walkers, ham_data, wave_data, trial, eps
    )

    return cc_cr0,cc_cr1,cc_cr2,cc_cr

def cc_impurity_solve(mf,mo_coeff,lo_coeff,eris=None,frozen=None):
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
    # log = logger.new_logger(mf if log is None else log)
    # cput1 = (logger.process_clock(), logger.perf_counter())

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
    _, nactocc, nactvir, _ = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]

    s1e = mf.get_ovlp() if eris is None else eris.s1e
    prjlo = fdot(lo_coeff.T, s1e, orbactocc)

    # solve impurity problem
    from pyscf.cc import CCSD
    mcc = CCSD(mf, mo_coeff=mo_coeff, frozen=frozen) #.set(verbose=verbose_imp)
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

    # MP2 fragment energy
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]

    # CCSD fragment energy
    mcc.kernel(eris=imp_eris, t1=t1, t2=t2)
    t1, t2 = mcc.t1, mcc.t2
    t2 += ccsd.einsum('ia,jb->ijab',t1,t1)
    ecorr_cc = ccsd.get_fragment_energy(oovv, t2, prjlo)
    t2 -= ccsd.einsum('ia,jb->ijab',t1,t1) 

    oovv = imp_eris = mcc = None
    # ci2 = ci2.transpose(0, 2, 1, 3)

    return ecorr_cc,t1,t2

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
                           prjlo=[],norb_frozen=[],ci1=None,ci2=None,
                           use_df_vecs=False,chol_cut=1e-6,
                           option_file='options.bin',
                           mo_file="mo_coeff.npz",
                           amp_file="amplitudes.npz",
                           chol_file="FCIDUMP_chol",
                           ):
    
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    # write ccsd amplitudes
    if isinstance(mf_cc, (CCSD, UCCSD)):
        mf = mf_cc._scf
        #cc = mf_cc
    else:
        mf = mf_cc

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
    if isinstance(norb_frozen, (int, float)) and norb_frozen == 0:
        norb_frozen = []
    elif isinstance(norb_frozen, int):
        norb_frozen = np.arange(norb_frozen)
    print(f'# frozen orbitals: {norb_frozen}')
    act = np.array([i for i in range(mol.nao) if i not in norb_frozen])
    print(f'# local active orbitals: {act}') #yichi
    print(f'# local active space size: {len(act)}') #yichi
    
    if getattr(mf, "with_df", None) is not None:
        if use_df_vecs:
            print('# use DF vectors as Cholesky vectors')
            _, chol, _, _ = pyscf_interface.generate_integrals(
                mol,mf.get_hcore(),mo_coeff[:,act],DFbas=mf.with_df.auxmol.basis)
        else:
            print("# composing ERIs from DF vectors")
            from pyscf import df
            chol_df = df.incore.cholesky_eri(mol, mf.with_df.auxmol.basis)
            chol_df = lib.unpack_tril(chol_df).reshape(chol_df.shape[0], -1)
            chol_df = chol_df.reshape((-1, mol.nao, mol.nao))
            eri_ao_df = lib.einsum('lpq,lrs->pqrs', chol_df, chol_df)
            #eri_df.reshape((nao*nao, nao*nao))
            print("# decomposing ERIs to Cholesky vectors")
            print(f"# Cholesky cutoff is: {chol_cut}")
            eri_mo_df = ao2mo.kernel(eri_ao_df,mo_coeff[:,act],compact=False)
            eri_mo_df = eri_mo_df.reshape(nbasis**2,nbasis**2)
            chol = pyscf_interface.modified_cholesky(eri_mo_df,max_error=chol_cut)
    else:
        eri_mo = ao2mo.kernel(mf.mol,mo_coeff[:,act],compact=False)
        chol = pyscf_interface.modified_cholesky(eri_mo,max_error=chol_cut)
    
    print("# Finished calculating Cholesky integrals\n")
    print('# Size of the correlation space')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {nbasis}')
    print(f'# Number of Cholesky vectors: {chol.shape[0]}\n')
    
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * lib.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    # overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        overlap = mf.get_ovlp(mol)
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
        q = np.eye(mol.nao- len(norb_frozen)) # in mo basis
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez(mo_file,mo_coeff=trial_coeffs,prjlo=prjlo)

    write_dqmc(mf.e_tot,h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,
               filename=chol_file,mo_coeffs=trial_coeffs)
    
    return None


def prep_lnoccsd_afqmc(options=None,prjlo=True,
                       full_cisd = False,
                       option_file="options.bin",
                       mo_file="mo_coeff.npz",
                       amp_file="amplitudes.npz",
                       chol_file="FCIDUMP_chol"):
    
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
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["ene0"] = options.get("ene0",0)
    options["use_gpu"] = options.get("use_gpu", False)

    if options["use_gpu"]:
        config.afqmc_config["use_gpu"] = True
    config.setup_jax()
    MPI = config.setup_comm()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
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
    wave_data.update({"ci1": ci1, "ci2": ci2})
    if full_cisd:
        full_ci1 = jnp.array(amplitudes["full_ci1"])
        full_ci2 = jnp.array(amplitudes["full_ci2"])
        no2mo_frzocc = jnp.array(amplitudes["no2mo_frzocc"])
        no2mo_act = jnp.array(amplitudes["no2mo_act"])
        wave_data.update(
            {"full_ci1":full_ci1,
             "full_ci2":full_ci2,
             "no2mo_frzocc":no2mo_frzocc,
             "no2mo_act":no2mo_act})
        
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
        print(f"# nchol: {nchol}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI

### the following part of the code ###
###   is to TEST the convergence   ###
###  run on the full active space  ###

@partial(jit, static_argnums=(3,))
def tot_hf_cr(walkers,ham_data:dict,wave_data:dict,trial:wavefunctions) -> jnp.ndarray:
    return vmap(_tot_hf_cr, in_axes=(None, None, 0, None, None))(
        ham_data['rot_h1'], ham_data['rot_chol'], walkers, trial, wave_data)

@partial(jit, static_argnums=(3,))
def _tot_hf_cr(rot_h1, rot_chol, walker, trial, wave_data):
    #m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    nocc = rot_h1.shape[0]
    _calc_green = wavefunctions.rhf(trial.norb,trial.nelec,n_batch=trial.n_batch)._calc_green
    green_walker = _calc_green(walker, wave_data)
    f = jnp.einsum('gij,jk->gik', rot_chol[:,:nocc,nocc:], green_walker.T[nocc:,:nocc], optimize='optimal')
    c = vmap(jnp.trace)(f)

    eneo2Jt = jnp.einsum('G,G->',c,c)*2 
    eneo2ext = jnp.einsum('Gxy,Gyx->',f,f)

    hf_corr = eneo2Jt - eneo2ext

    olp_ratio = _calc_olp_ratio_restricted(walker,wave_data)
    hf_cr = jnp.real(olp_ratio*hf_corr)

    return hf_cr

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

    # zero body
    h0_E0 = ham_data["h0"]-ham_data["E0"]
    mod_olp = _modified_ccsd_olp_restricted(walker,wave_data)

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

    ccsd_cr = jnp.real((h0_E0*mod_olp+d_overlap+jnp.sum(d_2_overlap)/2.0)/ccsd_olp)
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

        hf_cr = tot_hf_cr(
            prop_data["walkers"],ham_data,wave_data,trial)
        ccsd_cr = tot_ccsd_cr(
            prop_data["walkers"],ham_data,wave_data,trial,1e-5)
        e_corr = hf_cr + ccsd_cr

        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )
        blk_wt = jnp.sum(prop_data["weights"])
        blk_hf_cr = jnp.sum(hf_cr * prop_data["weights"]) / blk_wt
        blk_ccsd_cr = jnp.sum(ccsd_cr * prop_data["weights"]) / blk_wt
        blk_e_corr = jnp.sum(e_corr * prop_data["weights"]) / blk_wt
        blk_energy = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_energy
        )
        
        return prop_data,(blk_energy,blk_hf_cr,blk_ccsd_cr,blk_e_corr,blk_wt)

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

# @partial(jit, static_argnums=(2,3,5))
# def block_orb(prop_data: dict,
#               ham_data: dict,
#               prop: propagation.propagator,
#               trial: wave_function,
#               wave_data: dict,
#               sampler: sampling.sampler):
#         """Block scan function. Propagation and orbital_i energy calculation."""
#         prop_data["key"], subkey = random.split(prop_data["key"])
#         fields = random.normal(
#             subkey,
#             shape=(
#                 sampler.n_prop_steps,
#                 prop.n_walkers,
#                 ham_data["chol"].shape[0],
#             ),
#         )
#         # propgate n_prop_steps x dt
#         _step_scan_wrapper = lambda x, y: sampler._step_scan(
#             x, y, ham_data, prop, trial, wave_data
#         )
#         prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
#         prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
#             prop_data["weights"]
#         )
#         prop_data = prop.orthonormalize_walkers(prop_data)
#         prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

#         #olp_ratio = jnp.real(cal_olp_ratio(prop_data["walkers"], wave_data,trial))
#         # h0-E0 term #
#         #h0_e0 = ham_data["h0"]-ham_data["E0"]
#         #elec_orb_cr = jnp.real(frg_elec_cr(prop_data["walkers"],trial,wave_data,h0_e0))

#         hf_orb_cr,olp_ratio = frg_hf_cr(prop_data["walkers"],ham_data,wave_data,trial)
#         ccsd_orb_cr0,ccsd_orb_cr1,ccsd_orb_cr2,ccsd_orb_cr \
#             = frg_ccsd_cr(prop_data["walkers"], ham_data, 
#                                         wave_data, trial,1e-5)
#         energy_samples = jnp.real(
#             trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
#         )
#         energy_samples = jnp.where(
#             jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
#             prop_data["e_estimate"],
#             energy_samples,
#         )

#         blk_wt = jnp.sum(prop_data["weights"])
#         blk_energy = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
#         blk_hf_orb_cr = jnp.sum(hf_orb_cr * prop_data["weights"]) / blk_wt
#         blk_olp_ratio = jnp.sum(olp_ratio * prop_data["weights"]) / blk_wt
#         blk_ccsd_orb_cr0 = jnp.sum(ccsd_orb_cr0 * prop_data["weights"]) / blk_wt
#         blk_ccsd_orb_cr1 = jnp.sum(ccsd_orb_cr1 * prop_data["weights"]) / blk_wt
#         blk_ccsd_orb_cr2 = jnp.sum(ccsd_orb_cr2 * prop_data["weights"]) / blk_wt
#         blk_ccsd_orb_cr = jnp.sum(ccsd_orb_cr * prop_data["weights"]) / blk_wt
#         prop_data["pop_control_ene_shift"] = (
#             0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_energy
#         )
#         blk_orb_cr = blk_hf_orb_cr+blk_ccsd_orb_cr

#         return prop_data,(blk_energy,blk_wt,
#                           blk_hf_orb_cr,blk_olp_ratio,
#                           blk_ccsd_orb_cr0,blk_ccsd_orb_cr1,blk_ccsd_orb_cr2,
#                           blk_ccsd_orb_cr,blk_orb_cr)

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
        #h0_e0 = ham_data["h0"]-ham_data["E0"]
        #elec_orb_cr = jnp.real(frg_elec_cr(prop_data["walkers"],trial,wave_data,h0_e0))

        hf_orb_cr, hf_orb_en \
            = frg_hf_cr(prop_data["walkers"],ham_data,wave_data,trial)
        cc_orb_cr \
            = frg_ccsd_cr(prop_data["walkers"], ham_data, wave_data, trial,1e-5)
        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )

        blk_wt = jnp.sum(prop_data["weights"])
        blk_en = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
        blk_hf_orb_en = jnp.sum(hf_orb_en * prop_data["weights"]) / blk_wt
        blk_hf_orb_cr = jnp.sum(hf_orb_cr * prop_data["weights"]) / blk_wt
        blk_cc_orb_cr = jnp.sum(cc_orb_cr * prop_data["weights"]) / blk_wt
        blk_cc_orb_en = blk_hf_orb_cr+blk_cc_orb_cr
        
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_en
        )

        return prop_data,(blk_en,blk_wt,blk_hf_orb_en,blk_cc_orb_en)


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
    prop_data, (blk_en,blk_wt,blk_hf_orb_en,blk_cc_orb_en) \
        = lax.scan(
        _block_scan_wrapper, prop_data, xs=None, length=sampler.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

    return prop_data, (blk_en,blk_wt,blk_hf_orb_en,blk_cc_orb_en)

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
    prop_data,(blk_en,blk_wt,blk_hf_orb_en,blk_cc_orb_en) \
        = lax.scan(
        _sr_block_scan_wrapper, prop_data, xs=None, length=sampler.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sampler.n_sr_blocks * sampler.n_ene_blocks * prop.n_walkers
    )
    wt = jnp.sum(blk_wt)
    en = jnp.sum(blk_en * blk_wt) / wt
    hf_orb_en = jnp.sum(blk_hf_orb_en * blk_wt) / wt
    cc_orb_en = jnp.sum(blk_cc_orb_en * blk_wt) / wt

    return prop_data, (en,wt,hf_orb_en,cc_orb_en)

### debug mode ###
@partial(jit, static_argnums=(2,3,5))
def block_orb_dbg(prop_data: dict,
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

        blk_olp_r,blk_hf_orb_cr = frg_hf_cr_dbg(prop_data["walkers"],ham_data,wave_data,trial)
        blk_cc_orb_cr0,blk_cc_orb_cr1,blk_cc_orb_cr2,_ \
            = frg_cc_cr_dbg(prop_data["walkers"], ham_data, 
                                        wave_data, trial,1e-5)
        blk_en = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        blk_en = jnp.where(
            jnp.abs(blk_en - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],blk_en,
        )

        wt = jnp.sum(prop_data["weights"])
        en = jnp.sum(blk_en * prop_data["weights"]) / wt
        olp_r = jnp.sum(blk_olp_r * prop_data["weights"]) / wt
        hf_orb_cr = jnp.sum(blk_hf_orb_cr * prop_data["weights"]) / wt
        cc_orb_cr0 = jnp.sum(blk_cc_orb_cr0 * prop_data["weights"]) / wt
        cc_orb_cr1 = jnp.sum(blk_cc_orb_cr1 * prop_data["weights"]) / wt
        cc_orb_cr2 = jnp.sum(blk_cc_orb_cr2 * prop_data["weights"]) / wt
        cc_orb_cr = cc_orb_cr0 + cc_orb_cr1 + cc_orb_cr2
        orb_en = hf_orb_cr + cc_orb_cr

        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * en
        )

        return prop_data,(wt,en,olp_r,hf_orb_cr,cc_orb_cr0,
                          cc_orb_cr1,cc_orb_cr2,cc_orb_cr,orb_en)

@partial(jit, static_argnums=(2,3,5))
def _sr_block_scan_orb_dbg(
    prop_data: dict,
    ham_data: dict,
    prop: propagation.propagator,
    trial: wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    def _block_scan_wrapper(x,_):
        return block_orb_dbg(x,ham_data,prop,trial,wave_data,sampler)
    
    # propagate n_ene_blocks then do sr
    prop_data, (blk_wt,blk_en,blk_olp_r,blk_hf_orb_cr,
                blk_cc_orb_cr0,blk_cc_orb_cr1,blk_cc_orb_cr2,
                blk_cc_orb_cr,blk_orb_en) \
        = lax.scan(
        _block_scan_wrapper, prop_data, xs=None, length=sampler.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

    return prop_data, (blk_wt,blk_en,blk_olp_r,blk_hf_orb_cr,
                       blk_cc_orb_cr0,blk_cc_orb_cr1,blk_cc_orb_cr2,
                       blk_cc_orb_cr,blk_orb_en)

@partial(jit, static_argnums=(1, 3, 5))
def propagate_phaseless_orb_dbg(
    ham_data: dict,
    prop: propagation.propagator,
    prop_data: dict,
    trial: wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[jax.Array, dict]:
    def _sr_block_scan_wrapper(x,_):
        return _sr_block_scan_orb_dbg(x, ham_data, prop, trial, wave_data, sampler)

    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data,(blk_wt,blk_en,blk_olp_r,blk_hf_orb_cr,
               blk_cc_orb_cr0,blk_cc_orb_cr1,blk_cc_orb_cr2,
               blk_cc_orb_cr,blk_orb_en) \
        = lax.scan(
        _sr_block_scan_wrapper, prop_data, xs=None, length=sampler.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sampler.n_sr_blocks * sampler.n_ene_blocks * prop.n_walkers
    )

    wt = jnp.sum(blk_wt)
    en = jnp.sum(blk_en * blk_wt) / wt
    olp_r = jnp.sum(blk_olp_r * blk_wt) / wt
    hf_orb_cr = jnp.sum(blk_hf_orb_cr * blk_wt) / wt
    cc_orb_cr0 = jnp.sum(blk_cc_orb_cr0 * blk_wt) / wt
    cc_orb_cr1 = jnp.sum(blk_cc_orb_cr1 * blk_wt) / wt
    cc_orb_cr2 = jnp.sum(blk_cc_orb_cr2 * blk_wt) / wt
    cc_orb_cr = jnp.sum(blk_cc_orb_cr * blk_wt) / wt
    orb_en = jnp.sum(blk_orb_en * blk_wt) / wt

    return prop_data, (wt,en,olp_r,hf_orb_cr,cc_orb_cr0,
                       cc_orb_cr1,cc_orb_cr2,cc_orb_cr,orb_en)

import sys, os
from ad_afqmc.lno.cc import LNOCCSD
# from ad_afqmc.lno.afqmc import LNOAFQMC
from ad_afqmc.lno.base import lno
from pyscf.lib import logger
log = logger.Logger(sys.stdout, 6)

def get_fragment_energy(oovv, t2, prj):
    m = fdot(prj.T, prj)
    return lib.einsum('ijab,kjab,ik->',t2,2*oovv-oovv.transpose(0,1,3,2),m)

def get_mp2_frg_e(mf,frzfrag,eris,orbfragloc,can_orbfrag):

    mol = mf.mol
    nocc = mol.nelectron // 2 
    nao = mol.nao
    actfrag = np.array([i for i in range(nao) if i not in frzfrag])
    # frzocc = np.array([i for i in range(nocc) if i in frzfrag])
    actocc = np.array([i for i in range(nocc) if i in actfrag])
    actvir = np.array([i for i in range(nocc,nao) if i in actfrag])
    nactocc = len(actocc)
    nactocc = len(actocc)
    nactvir = len(actvir)

    s1e = eris.s1e
    can_prjlo = fdot(orbfragloc.T,s1e,can_orbfrag[:,actocc])
    mc = CCSD(mf, mo_coeff=can_orbfrag, frozen=frzfrag)
    mc.ao2mo = ccsd.ccsd_ao2mo.__get__(mc,mc.__class__)
    mc._s1e = s1e
    mc._h1e = eris.h1e
    mc._vhf = eris.vhf
    imp_eris = mc.ao2mo()
    if isinstance(imp_eris.ovov, np.ndarray):
        ovov = imp_eris.ovov
    else:
        ovov = imp_eris.ovov[()]
    oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
    ovov = None
    # MP2 fragment energy
    _, t2 = mc.init_amps(eris=imp_eris)[1:]
    emp2 = get_fragment_energy(oovv,t2,can_prjlo)
    
    return emp2

def get_eri_ao(mf):
    nao = mf.mol.nao
    if getattr(mf,"with_df",None) is not None:
        cderi = mf.with_df._cderi
        cderi = lib.unpack_tril(cderi).reshape(cderi.shape[0], -1)
        cderi = cderi.reshape((-1, nao, nao))
        eri_ao = lib.einsum('lpq,lrs->pqrs', cderi, cderi)
    else:
        eri_ao = mf._eri
        eri_ao = ao2mo.restore(1, eri_ao, nao)
    return eri_ao

def get_eri_mo(eri_ao,oa,va):
    eri_mo = lib.einsum('ip,aq,pqrs,rj,sb->iajb',oa.T,va.T,eri_ao,oa,va)
    return eri_mo

def make_lno(mfcc,orbfragloc,thresh_internal,thresh_external):

    if isinstance(thresh_external,(float,int)):
        thresh_ext_occ = thresh_ext_vir = thresh_external
    else:
        thresh_ext_occ, thresh_ext_vir  = thresh_external

    mf = mfcc._scf
    if getattr(mf,'with_df',None) is not None:
        print('Using DF integrals')
        eris = mfcc.ao2mo()
        s1e = eris.s1e if eris.s1e is None else mf.get_ovlp()
        fock = eris.fock  if eris.fock is None else mf.get_fock()
    else:
        print('Using true 4-index integrals')
        eris = None
        s1e = mf.get_ovlp()
        fock = mf.get_fock()
        
    nocc = np.count_nonzero(mf.mo_occ>1e-10)
    nmo = mf.mo_occ.size
    orbocc0, orbocc1, orbvir1, orbvir0 = mfcc.split_mo() # frz_occ, act_occ, act_vir, frz_vir
    _, moeocc1, moevir1, _ = mfcc.split_moe() # split energy

    lovir = abs(orbfragloc.T @ s1e @ orbvir1).max() > 1e-10
    m = fdot(orbfragloc.T, s1e, orbocc1) # overlap with all loc act_occs
    uocc1, uocc2 = lno.projection_construction(m, thresh_internal)
    moefragocc1, orbfragocc1 = lno.subspace_eigh(fock, fdot(orbocc1, uocc1))
    if lovir:
        m = fdot(orbfragloc.T, s1e, orbvir1)
        uvir1, uvir2 = lno.projection_construction(m, thresh_internal)
        moefragvir1, orbfragvir1 = lno.subspace_eigh(fock, fdot(orbvir1, uvir1))

    def moe_Ov(moefragocc):
        return (moefragocc[:,None] - moevir1).reshape(-1)

    eov = moe_Ov(moeocc1)
    # Construct PT2 dm_vv
    u = fdot(orbocc1.T, s1e, orbfragocc1)
    if getattr(mf,'with_df',None) is not None:
        print('Using DF integrals')
        ovov = eris.get_Ovov(u)
    else:
        print('Using true 4-index integrals')
        eri_ao = mf._eri
        nao = mf.mol.nao
        eri_ao = ao2mo.restore(1,eri_ao,nao)
        eri_mo = get_eri_mo(eri_ao,orbocc1,orbvir1)
        ovov = lib.einsum('iI,iajb->Iajb',u,eri_mo)
    eia = moe_Ov(moefragocc1)
    ejb = eov
    e1_or_e2 = 'e1'
    swapidx = 'ab'

    eiajb = (eia[:,None]+ejb).reshape(*ovov.shape)
    t2 = ovov / eiajb

    dmvv = lno.make_rdm1_mp2(t2, 'vv', e1_or_e2, swapidx)
   
    if lovir:
        dmvv = fdot(uvir2.T, dmvv, uvir2)

    # Construct PT2 dm_oo
    e1_or_e2 = 'e2'
    swapidx = 'ab'

    dmoo = lno.make_rdm1_mp2(t2, 'oo', e1_or_e2, swapidx)
    dmoo = fdot(uocc2.T, dmoo, uocc2)

    t2 = ovov = eiajb = None
    orbfragocc2, orbfragocc0 \
        = lno.natorb_compression(dmoo, orbocc1, thresh_ext_occ, uocc2)

    can_orbfragocc12 = lno.subspace_eigh(fock, np.hstack([orbfragocc2, orbfragocc1]))[1]
    orbfragocc12 = np.hstack([orbfragocc2, orbfragocc1])
    if lovir:
        orbfragvir2, orbfragvir0 \
            = lno.natorb_compression(dmvv,orbvir1,thresh_ext_vir,uvir2)

        can_orbfragvir12 = lno.subspace_eigh(fock, np.hstack([orbfragvir2, orbfragvir1]))[1]
        orbfragvir12 = np.hstack([orbfragvir2, orbfragvir1])
    else: 
        orbfragvir2, orbfragvir0 = lno.natorb_compression(dmvv,orbvir1,thresh_ext_vir)

        can_orbfragvir12 = lno.subspace_eigh(fock, orbfragvir2)[1]
        orbfragvir12 = orbfragvir2

    lno_orb = np.hstack([orbocc0, orbfragocc0, orbfragocc12,
                         orbfragvir12, orbfragvir0, orbvir0])
    can_orbfrag = np.hstack([orbocc0, orbfragocc0, can_orbfragocc12,
                        can_orbfragvir12, orbfragvir0, orbvir0])
    
    frzfrag = np.hstack([np.arange(orbocc0.shape[1]+orbfragocc0.shape[1]),
                         np.arange(nocc+orbfragvir12.shape[1],nmo)])

    return frzfrag, lno_orb , can_orbfrag


def run_lno_ccsd_afqmc(mfcc,thresh,frozen=None,options=None,
                       lo_type='boys',chol_cut=1e-6,nproc=None,
                       run_frg_list=None,
                       use_df_vecs=False,mp2=True,debug=False):
    '''
    mfcc: pyscf mean-field object
    thresh: lno thresh
    frozen: frozen orbitals
    options: afqmc options
    chol_cut: Cholesky Decomposition cutoff
    nproc: number of processors
    run_frg_list: list of the fragments to run
    '''

    from pyscf.cc.ccsd import CCSD
    from pyscf.cc.uccsd import UCCSD
    if isinstance(mfcc, (CCSD, UCCSD)):
        full_cisd = True
        mf = mfcc._scf
    else:
        full_cisd = False
        mf = mfcc

    # lo_type = 'pm'
    no_type = 'ie' # cim
    frag_lolist = '1o'

    _fdot = np.dot
    fdot = lambda *args: reduce(_fdot, args)

    if isinstance(thresh, list):
        thresh_occ, thresh_vir = thresh
    else:
        thresh_occ = thresh*10
        thresh_vir = thresh

    lno_cc = LNOCCSD(mf, thresh=thresh, frozen=frozen)
    lno_cc.thresh_occ = thresh_occ
    lno_cc.thresh_vir = thresh_vir
    lno_cc.lo_type = lo_type
    lno_cc.no_type = no_type
    lno_cc.frag_lolist = frag_lolist
    lno_cc.force_outcore_ao2mo = True

    s1e = lno_cc._scf.get_ovlp()
    orbactocc = lno_cc.split_mo()[1] # non-localized active occ
    # if localize:
    orbloc = lno_cc.get_lo(lo_type=lo_type) # localized active occ orbitals
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

    if frag_lolist == '1o':
        log.info('Using single-LO fragment') # this is what we use, every active local occ labels a fragment
        frag_lolist = [[i] for i in range(orbloc.shape[1])]
    else: print('Only support single LO fragment!')
    nfrag = len(frag_lolist)

    if not (no_type[0] in 'rei' and no_type[1] in 'rei'):
        log.warn('Input no_type "%s" is invalid.', no_type)
        raise ValueError

    frozen_mask = lno_cc.get_frozen_mask()
    thresh_pno = [thresh_occ,thresh_vir]
    print(f'# lno thresh {thresh_pno}')
    
    if run_frg_list is None:
        run_frg_list = range(nfrag)
    
    from jax import random
    seeds = random.randint(random.PRNGKey(options["seed"]),
                        shape=(len(run_frg_list),), minval=0, maxval=100*nfrag)
    
    for ifrag in run_frg_list:
        print(f'\n########### running fragment {ifrag+1} ##########')

        fraglo = frag_lolist[ifrag]
        orbfragloc = orbloc[:,fraglo] # the specific local active occ
        # frag_target_nocc, frag_target_nvir = frag_nonvlist[ifrag]
        THRESH_INTERNAL = 1e-10
        frzfrag, orbfrag, can_orbfrag \
            = make_lno(lno_cc, orbfragloc, THRESH_INTERNAL, thresh_pno)
        
        mol = mf.mol
        nocc = mol.nelectron // 2 
        nao = mol.nao
        actfrag = np.array([i for i in range(nao) if i not in frzfrag])
        # frzocc = np.array([i for i in range(nocc) if i in frzfrag])
        actocc = np.array([i for i in range(nocc) if i in actfrag])
        actvir = np.array([i for i in range(nocc,nao) if i in actfrag])
        nactocc = len(actocc)
        nactocc = len(actocc)
        nactvir = len(actvir)
        prjlo = fdot(orbfragloc.T,s1e,orbfrag[:,actocc])

        print(f'# active orbitals: {actfrag}')
        print(f'# active occupied orbitals: {actocc}')
        print(f'# active virtual orbitals: {actvir}')
        print(f'# frozen orbitals: {frzfrag}')

        if full_cisd:
            print('# Use full CCSD wavefunction')
            print('# This method is not size-extensive')
            print('# Projecting CI coefficients from MO to NO')
            frz_mo_idx = np.where(np.array(frozen_mask) == False)[0]
            act_mo_occ = np.array([i for i in range(nocc) if i not in frz_mo_idx])
            act_mo_vir = np.array([i for i in range(nocc,nao) if i not in frz_mo_idx])
            prj_no2mo = no2mo(mf.mo_coeff,s1e,orbfrag)
            prj_oo_act = prj_no2mo[np.ix_(act_mo_occ,actocc)]
            prj_vv_act = prj_no2mo[np.ix_(act_mo_vir,actvir)]
            full_t1 = mfcc.t1
            full_t2 = mfcc.t2
            # project to active no
            t1 = lib.einsum("ij,ia,ba->jb",prj_oo_act,full_t1,prj_vv_act.T)
            t2 = lib.einsum("ik,jl,ijab,db,ca->klcd",
                    prj_oo_act,prj_oo_act,full_t2,prj_vv_act.T,prj_vv_act.T)
            print('# Finished MO to NO projection')
            ecorr_ccsd = '  None  '
        else:
            ecorr_ccsd,t1,t2 = cc_impurity_solve(
                    mf,orbfrag,orbfragloc,frozen=frzfrag,eris=None
                    )
            ecorr_ccsd = f'{ecorr_ccsd:.8f}'
            print(f'# lno-ccsd fragment correlation energy: {ecorr_ccsd}')
 
        nelec_act = nactocc*2
        norb_act = nactocc+nactvir

        ci1 = np.array(t1)
        ci2 = t2 + lib.einsum("ia,jb->ijab",ci1,ci1)
        ci2 = ci2.transpose(0, 2, 1, 3)
        
        print(f'# number of active electrons: {nelec_act}')
        print(f'# number of active orbitals: {norb_act}')
        print(f'# number of frozen orbitals: {len(frzfrag)}')

        options["seed"] = seeds[ifrag]
        prep_lno_amp_chol_file(
            mf,orbfrag,options,
            norb_act=norb_act,nelec_act=nelec_act,
            prjlo=prjlo,norb_frozen=frzfrag,
            ci1=ci1,ci2=ci2,use_df_vecs=use_df_vecs,
            chol_cut=chol_cut,
            )
        
        #MP2 correction 
        if mp2:
            print('# running fragment MP2')
            can_prjlo = fdot(orbfragloc.T,s1e,can_orbfrag[:,actocc])

            from pyscf.cc import CCSD
            mcc = CCSD(mf, mo_coeff=can_orbfrag, frozen=frzfrag)
            mcc.ao2mo = ccsd.ccsd_ao2mo.__get__(mcc, mcc.__class__)
            mcc._s1e = s1e
            imp_eris = mcc.ao2mo()
            if isinstance(imp_eris.ovov, np.ndarray):
                ovov = imp_eris.ovov
            else:
                ovov = imp_eris.ovov[()]
            oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
            ovov = None

            _, t2 = mcc.init_amps(eris=imp_eris)[1:]
            ecorr_pt2 = get_fragment_energy(oovv, t2, can_prjlo)
            ecorr_pt2 = f'{ecorr_pt2:.8f}'
            print(f'# lno-mp2 fragment energy: {ecorr_pt2}')
        else:
            ecorr_pt2 = '  None  '

        # Run AFQMC
        use_gpu = options["use_gpu"]
        if use_gpu:
            print(f'# running AFQMC on GPU')
            gpu_flag = "--use_gpu"
            mpi_prefix = ""
            nproc = None
            from mpi4py import MPI
            if not MPI.Is_finalized():
                MPI.Finalize() # CCSD initializes MPI
        else:
            print(f'# running AFQMC on CPU')
            gpu_flag = ""
            mpi_prefix = "mpirun "
            if nproc is not None:
                mpi_prefix += f"-np {nproc} "
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)   
        if debug:
            script = f"{dir_path}/run_lnocc_frg_dbg.py"
        else:
            script = f"{dir_path}/run_lnocc_frg.py"
        os.system(
            f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
            f"{mpi_prefix} python {script} {gpu_flag} |tee frg_{ifrag+1}.out"
        )

        with open(f'frg_{ifrag+1}.out', 'a') as out_file:
            # if mp2:
            print(f"lno-mp2 orb_corr: {ecorr_pt2}",file=out_file)
            # if not full_cisd: 
            print(f"lno-ccsd orb_corr: {ecorr_ccsd}",file=out_file)
            print(f"number of active electrons: {nelec_act}",file=out_file)
            print(f"number of active orbitals: {norb_act}",file=out_file)

    from pyscf import mp
    mmp = mp.MP2(mf, frozen=frozen)
    e_mp2 = mmp.kernel()[0]

    if debug:
        frg2result_dbg(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2)
    else:
        frg2result(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2)

    return None


#### results analysis ####
def frg2result(lno_thresh,nfrag,e_mf,e_mp2):
    with open('results.out', 'w') as out_file:
        print('# frag  mp2_orb_corr  ccsd_orb_corr' \
              '  afqmc_hf_orb_en  err  afqmc_cc_orb_en  err' \
              '  norb  nelec  time',file=out_file)
        for ifrag in range(nfrag):
            with open(f"frg_{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "lno-mp2 orb_corr" in line:
                        e_mp2_orb_en = line.split()[2]
                    if "lno-ccsd orb_corr" in line:
                        e_ccsd_orb_en = line.split()[2]
                    if "lno-ccsd-afqmc hf_orb_en" in line:
                        hf_orb_en = line.split()[2]
                        hf_orb_en_err = line.split()[4]
                    if "lno-ccsd-afqmc cc_orb_en" in line:
                        cc_orb_en = line.split()[2]
                        cc_orb_en_err = line.split()[4]
                    if "number of active orbitals" in line:
                        norb = line.split()[4]
                    if "number of active electrons" in line:
                        nelec = line.split()[4]
                    if "total run time" in line:
                        tot_time = line.split()[3]
                print(f'{ifrag+1:3d}  '
                      f'{e_mp2_orb_en}  {e_ccsd_orb_en}  '
                      f'{hf_orb_en}  {hf_orb_en_err}  '
                      f'{cc_orb_en}  {cc_orb_en_err}  '
                      f'{norb}  {nelec}  {tot_time}', file=out_file)

    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == 'None'] = '0.000000' 

    data = np.array(data.reshape(nfrag,10))
    e_mp2_orb_en = np.array(data[:,1],dtype='float32')
    e_ccsd_orb_en = np.array(data[:,2],dtype='float32')
    hf_orb_en = np.array(data[:,3],dtype='float32')
    hf_orb_en_err = np.array(data[:,4],dtype='float32')
    cc_orb_en = np.array(data[:,5],dtype='float32')
    cc_orb_en_err = np.array(data[:,6],dtype='float32')
    norb = np.array(data[:,7],dtype='int32')
    nelec = np.array(data[:,8],dtype='int32')
    tot_time = np.array(data[:,9],dtype='float32')

    e_mp2_corr = sum(e_mp2_orb_en)
    mp2_cr = e_mp2 - e_mp2_corr
    e_ccsd_corr = sum(e_ccsd_orb_en)
    afqmc_hf_corr = sum(hf_orb_en)
    afqmc_hf_corr_err = np.sqrt(sum(hf_orb_en_err**2))
    afqmc_cc_corr = sum(cc_orb_en)
    afqmc_cc_corr_err = np.sqrt(sum(cc_orb_en_err**2))
    norb_avg = np.mean(norb)
    nelec_avg = np.mean(nelec)
    norb_max = max(norb)
    nelec_max = max(nelec)
    tot_time = sum(tot_time)

    e_mp2_corr = f'{e_mp2_corr:.6f}'
    e_ccsd_corr = f'{e_ccsd_corr:.6f}'
    afqmc_hf_corr = f'{afqmc_hf_corr:.6f}'
    afqmc_hf_corr_err = f'{afqmc_hf_corr_err:.6f}'
    afqmc_cc_corr = f'{afqmc_cc_corr:.6f}'
    afqmc_cc_corr_err = f'{afqmc_cc_corr_err:.6f}'

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# mean-field energy: {e_mf:.8f}\n')
        out_file.write(f'# lno-thresh {lno_thresh}\n')
        out_file.write(f'# e_mp2_corr: {e_mp2_corr}\n')
        out_file.write(f'# e_ccsd_corr: {e_ccsd_corr}\n')
        out_file.write(f'# afqmc/hf_corr: {afqmc_hf_corr} +/- {afqmc_hf_corr_err}\n')
        out_file.write(f'# afqmc/cc_corr: {afqmc_cc_corr} +/- {afqmc_cc_corr_err}\n')
        out_file.write(f'# mp2_correction: {mp2_cr:.8f}\n')
        out_file.write(f'# number of orbitals: average {norb_avg:.2f} maxium {norb_max}\n')
        out_file.write(f'# number of electrons: average {nelec_avg:.2f} maxium {nelec_max}\n')
        out_file.write(f'# total run time: {tot_time:.2f}\n')
    
    return None

def frg2result_dbg(lno_thresh,nfrag,e_mf,e_mp2):
    with open('results.out', 'w') as out_file:
        print('# frag  mp2_orb_corr  ccsd_orb_corr' \
              '  olp_ratio  err  qmc_hf_orb_cr  err'
              '  qmc_cc_orb_cr0  err  qmc_cc_orb_cr1  err' \
              '  qmc_cc_orb_cr2  err  qmc_cc_orb_cr  err'
              '  qmc_orb_en  err  nelec  norb  time',file=out_file)
        for ifrag in range(nfrag):
            # ccsd_orb_en = None
            with open(f"frg_{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "lno-mp2 orb_corr:" in line:
                        mp2_orb_en = line.split()[2]
                    if "lno-ccsd orb_corr:" in line:
                        ccsd_orb_en = line.split()[2]
                    if "lno-afqmc/cc olp_r:" in line:
                        olp_r = line.split()[2]
                        olp_r_err = line.split()[4]
                    if "lno-afqmc/cc hf_orb_cr:" in line:
                        hf_orb_cr = line.split()[2]
                        hf_orb_cr_err = line.split()[4]
                    if "lno-afqmc/cc cc_orb_cr0:" in line:
                        cc_orb_cr0 = line.split()[2]
                        cc_orb_cr0_err = line.split()[4]
                    if "lno-afqmc/cc cc_orb_cr1:" in line:
                        cc_orb_cr1 = line.split()[2]
                        cc_orb_cr1_err = line.split()[4]
                    if "lno-afqmc/cc cc_orb_cr2:" in line:
                        cc_orb_cr2 = line.split()[2]
                        cc_orb_cr2_err = line.split()[4]
                    if "lno-afqmc/cc cc_orb_cr:" in line:
                        cc_orb_cr = line.split()[2]
                        cc_orb_cr_err = line.split()[4]
                    if "lno-afqmc/cc orb_en:" in line:
                        qmc_orb_en = line.split()[2]
                        qmc_orb_en_err = line.split()[4]
                    if "number of active electrons:" in line:
                        nelec = line.split()[4]
                    if "number of active orbitals:" in line:
                        norb = line.split()[4]
                    if "total run time" in line:
                        tot_time = line.split()[3]
                if ccsd_orb_en is None:
                    ccsd_orb_en = '  None  '
                print(f'{ifrag+1:3d}  '
                      f'{mp2_orb_en}  {ccsd_orb_en}  '
                      f'{olp_r}  {olp_r_err}  {hf_orb_cr}  {hf_orb_cr_err}  '
                      f'{cc_orb_cr0}  {cc_orb_cr0_err}  {cc_orb_cr1}  {cc_orb_cr1_err}  '
                      f'{cc_orb_cr2}  {cc_orb_cr2_err}  {cc_orb_cr}  {cc_orb_cr_err}  '
                      f'{qmc_orb_en}  {qmc_orb_en_err}  {nelec}  {norb}  {tot_time}  ',
                      file=out_file)

    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == 'None'] = '0.000000' 

    data = np.array(data.reshape(nfrag,20))
    mp2_orb_en = np.array(data[:,1],dtype='float32')
    ccsd_orb_en = np.array(data[:,2],dtype='float32')
    olp_r = np.array(data[:,3],dtype='float32')
    olp_r_err = np.array(data[:,4],dtype='float32')
    hf_orb_cr = np.array(data[:,5],dtype='float32')
    hf_orb_cr_err = np.array(data[:,6],dtype='float32')
    cc_orb_cr0 = np.array(data[:,7],dtype='float32')
    cc_orb_cr0_err = np.array(data[:,8],dtype='float32')
    cc_orb_cr1 = np.array(data[:,9],dtype='float32')
    cc_orb_cr1_err = np.array(data[:,10],dtype='float32')
    cc_orb_cr2 = np.array(data[:,11],dtype='float32')
    cc_orb_cr2_err = np.array(data[:,12],dtype='float32')
    cc_orb_cr = np.array(data[:,13],dtype='float32')
    cc_orb_cr_err = np.array(data[:,14],dtype='float32')
    qmc_orb_en = np.array(data[:,15],dtype='float32')
    qmc_orb_en_err = np.array(data[:,16],dtype='float32')
    nelec = np.array(data[:,17],dtype='int32')
    norb = np.array(data[:,18],dtype='int32')
    tot_time = np.array(data[:,19],dtype='float32')

    mp2_corr = sum(mp2_orb_en)
    mp2_cr = e_mp2 - mp2_corr
    ccsd_corr = sum(ccsd_orb_en)
    olp_r = np.mean(olp_r)
    olp_r_err = np.sqrt(sum(olp_r_err**2))/len(olp_r_err)
    qmc_hf_cr = sum(hf_orb_cr)
    qmc_hf_cr_err = np.sqrt(sum(hf_orb_cr_err**2))
    qmc_cc_cr0 = sum(cc_orb_cr0)
    qmc_cc_cr0_err = np.sqrt(sum(cc_orb_cr0_err**2))
    qmc_cc_cr1 = sum(cc_orb_cr1)
    qmc_cc_cr1_err = np.sqrt(sum(cc_orb_cr1_err**2))
    qmc_cc_cr2 = sum(cc_orb_cr2)
    qmc_cc_cr2_err = np.sqrt(sum(cc_orb_cr2_err**2))
    qmc_cc_cr = sum(cc_orb_cr)
    qmc_cc_cr_err = np.sqrt(sum(cc_orb_cr_err**2))
    qmc_corr = sum(qmc_orb_en)
    qmc_corr_err = np.sqrt(sum(qmc_orb_en_err**2))
    nelec_avg = np.mean(nelec)
    norb_avg = np.mean(norb)
    nelec_max = max(nelec)
    norb_max = max(norb)
    tot_time = sum(tot_time)

    mp2_corr = f'{mp2_corr:.6f}'
    ccsd_corr = f'{ccsd_corr:.6f}'
    olp_r = f'{olp_r:.6f}'
    olp_r_err = f'{olp_r_err:.6f}'
    qmc_hf_cr = f'{qmc_hf_cr:.6f}'
    qmc_hf_cr_err = f'{qmc_hf_cr_err:.6f}'
    qmc_cc_cr0 = f'{qmc_cc_cr0:.6f}'
    qmc_cc_cr0_err = f'{qmc_cc_cr0_err:.6f}'
    qmc_cc_cr1 = f'{qmc_cc_cr1:.6f}'
    qmc_cc_cr1_err = f'{qmc_cc_cr1_err:.6f}'
    qmc_cc_cr2 = f'{qmc_cc_cr2:.6f}'
    qmc_cc_cr2_err = f'{qmc_cc_cr2_err:.6f}'
    qmc_cc_cr = f'{qmc_cc_cr:.6f}'
    qmc_cc_cr_err = f'{qmc_cc_cr_err:.6f}'
    qmc_corr = f'{qmc_corr:.6f}'
    qmc_corr_err = f'{qmc_corr_err:.6f}'

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# mean-field energy: {e_mf:.8f}\n')
        out_file.write(f'# lno-thresh {lno_thresh}\n')
        out_file.write(f'# lno-mp2_corr: {mp2_corr}\n')
        out_file.write(f'# lno-ccsd_corr: {ccsd_corr}\n')
        out_file.write(f'# lno-afqmc/cc olp_r: {olp_r} +/- {olp_r_err}\n')
        out_file.write(f'# lno-afqmc/cc hf_cr: {qmc_hf_cr} +/- {qmc_hf_cr_err}\n')
        out_file.write(f'# lno-afqmc/cc cc_cr0: {qmc_cc_cr0} +/- {qmc_cc_cr0_err}\n')
        out_file.write(f'# lno-afqmc/cc cc_cr1: {qmc_cc_cr1} +/- {qmc_cc_cr1_err}\n')
        out_file.write(f'# lno-afqmc/cc cc_cr2: {qmc_cc_cr2} +/- {qmc_cc_cr2_err}\n')
        out_file.write(f'# lno-afqmc/cc cc_cr: {qmc_cc_cr} +/- {qmc_cc_cr_err}\n')
        out_file.write(f'# lno-afqmc/cc corr: {qmc_corr} +/- {qmc_corr_err}\n')
        out_file.write(f'# mp2_correction: {mp2_cr:.8f}\n')
        out_file.write(f'# number of electrons: average {nelec_avg:.2f} maxium {nelec_max}\n')
        out_file.write(f'# number of orbitals: average {norb_avg:.2f} maxium {norb_max}\n')
        out_file.write(f'# total run time: {tot_time:.2f}\n')
    
    return None

def sum_results(n_results):
    with open('sum_results.out', 'w') as out_file:
        print("# lno-thresh(occ,vir) "
              "  mp2_corr  ccsd_corr"
              "  qmc/hf_corr   err "
              "  qmc/ccsd_corr   err "
              "  mp2_cr nelec_avg   nelec_max  "
              "  norb_avg   norb_max  "
              "  run_time",file=out_file)
        for i in range(n_results):
            with open(f"results.out{i+1}","r") as read_file:
                for line in read_file:
                    if "lno-thresh" in line:
                        thresh_occ = line.split()[-2]
                        thresh_vir = line.split()[-1]
                        thresh_occ = float(thresh_occ.strip('()[],'))
                        thresh_vir = float(thresh_vir.strip('()[],'))
                    if "e_mp2_corr:" in line:
                        mp2_corr = line.split()[-1]
                    if "e_ccsd_corr:" in line:
                        ccsd_corr = line.split()[-1]
                    if "afqmc/hf_corr:" in line:
                        afqmc_hf_corr = line.split()[-3]
                        afqmc_hf_corr_err = line.split()[-1]
                    if "afqmc/cc_corr:" in line:
                        afqmc_cc_corr = line.split()[-3]
                        afqmc_cc_corr_err = line.split()[-1]
                    if "mp2_correction:" in line:
                        mp2_cr = line.split()[-1]
                    if "electrons:" in line:
                        nelec_avg = line.split()[-3]
                        nelec_max = line.split()[-1]
                    if "orbitals:" in line:
                        norb_avg = line.split()[-3]
                        norb_max = line.split()[-1]
                    if "time:" in line:
                        run_time = line.split()[-1]
            print(f" ({thresh_occ:.2e},{thresh_vir:.2e}) \t"
                  f" {mp2_corr} \t {ccsd_corr} \t"
                  f" {afqmc_hf_corr} \t {afqmc_hf_corr_err} \t"
                  f" {afqmc_cc_corr} \t {afqmc_cc_corr_err} \t"
                  f" {mp2_cr}  {nelec_avg} \t {nelec_max} \t"
                  f" {norb_avg}  \t {norb_max} \t {run_time}",file=out_file)
    return None

def sum_results_dbg(n_results):
    with open('sum_results.out', 'w') as out_file:
        print("# thresh(occ,vir) "
              "  mp2_corr  ccsd_corr  olp_ratio  err"
              "  qmc_hf_cr  err  qmc_cc_cr0   err"
              "  qmc_cc_cr1   err  qmc_cc_cr2   err"
              "  qmc_cc_cr   err  qmc_corr   err  mp2_correction"
              "  nelec_avg   nelec_max  norb_avg   norb_max  "
              "  run_time",file=out_file)
        for i in range(n_results):
            with open(f"results.out{i+1}","r") as read_file:
                for line in read_file:
                    if "lno-thresh" in line:
                        thresh_occ = line.split()[-2]
                        thresh_vir = line.split()[-1]
                        thresh_occ = float(thresh_occ.strip('()[],'))
                        thresh_vir = float(thresh_vir.strip('()[],'))
                    if "lno-mp2_corr:" in line:
                        mp2_corr = line.split()[-1]
                    if "lno-ccsd_corr:" in line:
                        ccsd_corr = line.split()[-1]
                    if "lno-afqmc/cc olp_r:" in line:
                        olp_r = line.split()[-3]
                        olp_r_err = line.split()[-1]
                    if "lno-afqmc/cc hf_cr:" in line:
                        qmc_hf_cr = line.split()[-3]
                        qmc_hf_cr_err = line.split()[-1]
                    if "lno-afqmc/cc cc_cr0:" in line:
                        qmc_cc_cr0 = line.split()[-3]
                        qmc_cc_cr0_err = line.split()[-1]
                    if "lno-afqmc/cc cc_cr1:" in line:
                        qmc_cc_cr1 = line.split()[-3]
                        qmc_cc_cr1_err = line.split()[-1]
                    if "lno-afqmc/cc cc_cr2:" in line:
                        qmc_cc_cr2 = line.split()[-3]
                        qmc_cc_cr2_err = line.split()[-1]
                    if "lno-afqmc/cc cc_cr:" in line:
                        qmc_cc_cr = line.split()[-3]
                        qmc_cc_cr_err = line.split()[-1]
                    if "lno-afqmc/cc corr:" in line:
                        qmc_corr = line.split()[-3]
                        qmc_corr_err = line.split()[-1]
                    if "mp2_correction:" in line:
                        mp2_cr = line.split()[-1]
                    if "electrons:" in line:
                        nelec_avg = line.split()[-3]
                        nelec_max = line.split()[-1]
                    if "orbitals:" in line:
                        norb_avg = line.split()[-3]
                        norb_max = line.split()[-1]
                    if "time:" in line:
                        run_time = line.split()[-1]
            print(f"  ({thresh_occ:.2e},{thresh_vir:.2e})"
                  f"  {mp2_corr}  {ccsd_corr}  {olp_r}  {olp_r_err}"
                  f"  {qmc_hf_cr}  {qmc_hf_cr_err}"
                  f"  {qmc_cc_cr0}  {qmc_cc_cr0_err}"
                  f"  {qmc_cc_cr1}  {qmc_cc_cr1_err}"
                  f"  {qmc_cc_cr2}  {qmc_cc_cr2_err}"
                  f"  {qmc_cc_cr}  {qmc_cc_cr_err}"
                  f"  {qmc_corr} \t {qmc_corr_err} \t"
                  f"  {mp2_cr}  {nelec_avg} \t {nelec_max} \t"
                  f"  {norb_avg}  \t {norb_max} \t {run_time}",
                  file=out_file)
    return None

def lno_data(data):
      new_data = []
      lines = data.splitlines()
      for line in lines:
            columns = line.split()
            if len(columns)>1:
                  if not line.startswith("#"): 
                        new_data.append(columns)

      new_data = np.array(new_data)

      lno_thresh = []
      for i in range(new_data.shape[0]):
            thresh_vir = new_data[:,0][i].split(sep=',')[1]
            thresh_vir = float(thresh_vir.strip('(),'))
            lno_thresh.append(thresh_vir)

      lno_data = np.array(new_data[:,1:],dtype="float32")

      lno_thresh = np.array(lno_thresh,dtype="float32")
      # lno_thresh[-1] = 1e-10 # last thresh = 0.0
      lno_afqmc_corr = lno_data[:,0]
      lno_afqmc_err = lno_data[:,1]
      lno_afqmc_mp2_corr = lno_data[:,2]
      lno_ccsd_corr = lno_data[:,3]

      lno_mp2_cr = lno_afqmc_mp2_corr-lno_afqmc_corr
      lno_ccsd_mp2_corr = lno_ccsd_corr+lno_mp2_cr

      return lno_thresh,lno_afqmc_corr,lno_afqmc_mp2_corr,lno_afqmc_err,lno_ccsd_corr,lno_ccsd_mp2_corr

def lno_data_dbg(data):
      new_data = []
      lines = data.splitlines()
      for line in lines:
            columns = line.split()
            if len(columns)>1:
                  if not line.startswith("#"): 
                        new_data.append(columns)

      new_data = np.array(new_data)

      lno_thresh = []
      for i in range(new_data.shape[0]):
            thresh_vir = new_data[:,0][i].split(sep=',')[1]
            thresh_vir = float(thresh_vir.strip('(),'))
            lno_thresh.append(thresh_vir)

      lno_data = np.array(new_data[:,1:],dtype="float32")

      lno_thresh = np.array(lno_thresh,dtype="float32")
      mp2_corr = lno_data[:,0]
      ccsd_corr = lno_data[:,1]
      olp_r = lno_data[:,2]
      olp_r_err = lno_data[:,3]
      qmc_hf_cr = lno_data[:,4]
      qmc_hf_cr_err = lno_data[:,5]
      qmc_cc_cr0 = lno_data[:,6]
      qmc_cc_cr0_err = lno_data[:,7]
      qmc_cc_cr1 = lno_data[:,8]
      qmc_cc_cr1_err = lno_data[:,9]
      qmc_cc_cr2 = lno_data[:,10]
      qmc_cc_cr2_err = lno_data[:,11]
      qmc_cc_cr = lno_data[:,12]
      qmc_cc_cr_err = lno_data[:,13]
      qmc_corr = lno_data[:,14]
      qmc_corr_err = lno_data[:,15]
      mp2_cr = lno_data[:,16]
      nelec_avg = lno_data[:,17]
      nelec_max = lno_data[:,18]
      norb_avg = lno_data[:,19]
      norb_max = lno_data[:,20]
      time = lno_data[:,21]

      return (lno_thresh,mp2_corr,ccsd_corr,olp_r,olp_r_err,
              qmc_hf_cr,qmc_hf_cr_err,qmc_cc_cr0,qmc_cc_cr0_err,
              qmc_cc_cr1,qmc_cc_cr1_err,qmc_cc_cr2,qmc_cc_cr2_err,
              qmc_cc_cr,qmc_cc_cr_err,qmc_corr,qmc_corr_err,mp2_cr,
              nelec_avg,nelec_max,norb_avg,norb_max,time)