from functools import partial
import jax
from jax import numpy as jnp
from jax import lax, vmap, jvp, random, jit
from typing import Tuple
from ad_afqmc import propagation, sampling, wavefunctions

@jax.jit
def _calc_olp_ratio_restricted(walker: jax.Array, wave_data: dict) -> complex:
    '''
    <psi_hf|walker>/<psi_cisd|walker>
    '''
    nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
    GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    #o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
    o1 = jnp.einsum("ia,ia->", ci1, GF[:, nocc:])
    o2 = 2 * jnp.einsum("iajb, ia, jb->", ci2, GF[:, nocc:], GF[:, nocc:]) \
        - jnp.einsum("iajb, ib, ja->", ci2, GF[:, nocc:], GF[:, nocc:])
    return 1/(1.0 + 2 * o1 + o2)

@jax.jit
def _frg_mod_ci_olp_restricted(walker: jax.Array, wave_data: dict) -> complex:
    '''
    <psi_ccsd|walker>=<psi_0|walker>+C_ia^*G_ia+C_iajb^*(G_iaG_jb-G_ibG_ja)
    modified CISD overlap returns the second and the third term
    that is, the overlap of the walker with the CCSD wavefunction
    without the hartree-fock part
    and skip one sum over the occ
    '''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
    gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    o1 = jnp.einsum("ia,ka,ik->", ci1, gf[:, nocc:],m)
    o2 = 2 * jnp.einsum("iajb,ka,jb,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m) \
        - jnp.einsum("iajb,kb,ja,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m)
    olp = (2*o1+o2)*o0
    return olp

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
###############################################################################

@jax.jit
def _thouless_linear(t,walker):
    new_walker = walker + t.dot(walker)
    return new_walker

@jax.jit
def _frg_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    <psi_ccsd|exp(x*h1_mod)|walker>/<psi_ccsd|walker> without the hf part
    '''
    walker_1x = _thouless_linear(x*h1_mod, walker)
    olp = _frg_mod_ci_olp_restricted(walker_1x, wave_data)
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
    olp = _frg_mod_ci_olp_restricted(walker_2x, wave_data)
    return olp

@partial(jit, static_argnums=(3,))
def _frg_hf_eorb(rot_h1, rot_chol, walker, trial, wave_data):
    '''hf orbital correlation energy'''
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
    # olp_ratio = _calc_olp_ratio_restricted(walker,wave_data)
    # hf_orb_cr = jnp.real(olp_ratio*hf_orb_en)
    return jnp.real(hf_orb_en)

@partial(jit, static_argnums=(3,))
def _frg_hf_cr(walker, ham_data, wave_data, trial):
    '''hf orbital correlation energy multiplies the overlap ratio'''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    rot_h1, rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
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

@partial(jit, static_argnums=(3,))
def frg_hf_cr(walkers,ham_data,wave_data,trial):
    hf_orb_cr = vmap(_frg_hf_cr, in_axes=(0, None, None, None))(
        walkers, ham_data, wave_data, trial)
    return hf_orb_cr

@partial(jit, static_argnums=(3,))
def _frg_ci_cr(
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
    mod_olp = _frg_mod_ci_olp_restricted(walker,wave_data)

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

    ccsd_cr = jnp.real(
        (h0_E0*mod_olp + d_overlap + jnp.sum(d_2_overlap) / 2.0) / ccsd_olp)

    return ccsd_cr #ccsd_cr0,ccsd_cr1,ccsd_cr2,

@partial(jit, static_argnums=(3,))
def frg_ci_cr(
        walkers: jax.Array,
        ham_data: dict,
        wave_data: dict,
        trial: wavefunctions,
        eps :float = 1e-5) -> jax.Array:
    
    ccsd_cr = vmap(_frg_ci_cr, 
                   in_axes=(0, None, None, None, None))(
                       walkers, ham_data, wave_data, trial, eps)

    return ccsd_cr #ccsd_cr0,ccsd_cr1,ccsd_cr2

@partial(jit, static_argnums=(3,))
def _orb_energy(
        walker: jax.Array,
        ham_data: dict,
        wave_data: dict,
        trial: wavefunctions) -> jax.Array:
    
    e_orb = _frg_hf_cr(walker, ham_data, wave_data, trial) \
         + _frg_ci_cr(walker, ham_data, wave_data, trial, 1e-5)
    
    return e_orb

@partial(jit, static_argnums=(3,))
def orb_energy(
        walkers: jax.Array,
        ham_data: dict,
        wave_data: dict,
        trial: wavefunctions) -> jax.Array:
    
    e_orb = vmap(_orb_energy, 
                   in_axes=(0, None, None, None))(
                       walkers, ham_data, wave_data, trial)

    return e_orb

@partial(jit, static_argnums=(2,3,5))
def block_orb(prop_data: dict,
              ham_data: dict,
              prop: propagation.propagator,
              trial: wavefunctions,
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

        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )
        eorb = orb_energy(prop_data["walkers"], ham_data, wave_data, trial)

        blk_wt = jnp.sum(prop_data["weights"])
        blk_en = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
        blk_eorb = jnp.sum(eorb * prop_data["weights"]) / blk_wt
        
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_en
        )

        return prop_data,(blk_wt,blk_en,blk_eorb)

@partial(jit, static_argnums=(2,3,5))
def _sr_block_scan_orb(
    prop_data: dict,
    ham_data: dict,
    prop: propagation.propagator,
    trial: wavefunctions,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    def _block_scan_wrapper(x,_):
        return block_orb(x,ham_data,prop,trial,wave_data,sampler)
    
    # propagate n_ene_blocks then do sr
    prop_data,(blk_wt,blk_en,blk_eorb) \
        = lax.scan(
        _block_scan_wrapper, prop_data, xs=None, length=sampler.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

    return prop_data,(blk_wt,blk_en,blk_eorb)

@partial(jit, static_argnums=(1, 3, 5))
def propagate_phaseless_orb(
    ham_data: dict,
    prop: propagation.propagator,
    prop_data: dict,
    trial: wavefunctions,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[jax.Array, dict]:
    def _sr_block_scan_wrapper(x,_):
        return _sr_block_scan_orb(x, ham_data, prop, trial, wave_data, sampler)

    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data,(blk_wt,blk_en,blk_eorb) \
        = lax.scan(
        _sr_block_scan_wrapper, prop_data, xs=None, length=sampler.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sampler.n_sr_blocks * sampler.n_ene_blocks * prop.n_walkers
    )

    wt = jnp.sum(blk_wt)
    en = jnp.sum(blk_en * blk_wt) / wt
    eorb = jnp.sum(blk_eorb * blk_wt) / wt

    return prop_data,(wt,en,eorb)

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
def _frg_ci_cr_dbg(
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
    mod_olp = _frg_mod_ci_olp_restricted(walker,wave_data)

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
def frg_ci_cr_dbg(
        walkers: jax.Array,
        ham_data: dict,
        wave_data: dict,
        trial: wavefunctions,
        eps :float = 1e-5) -> jax.Array:
    
    cc_cr0,cc_cr1,cc_cr2,cc_cr \
        = vmap(_frg_ci_cr_dbg, in_axes=(0, None, None, None, None))(
        walkers, ham_data, wave_data, trial, eps
    )

    return cc_cr0,cc_cr1,cc_cr2,cc_cr

@partial(jit, static_argnums=(2,3,5))
def block_orb_dbg(prop_data: dict,
              ham_data: dict,
              prop: propagation.propagator,
              trial: wavefunctions,
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
            = frg_ci_cr_dbg(prop_data["walkers"], ham_data, 
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
    trial: wavefunctions,
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
    trial: wavefunctions,
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
### debug mode ends ###