import jax
from jax import lax, jit, jvp, vmap
from jax import numpy as jnp
from functools import partial

### e0 part ###
@jax.jit
def _hf_walker_olp(walker):
    nocc = walker.shape[1]
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    return o0

@jax.jit
def _hf_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array) -> complex:
    '''
    <psi_0|exp(x*h1_mod)|walker>
    '''
    t = x * h1_mod
    walker_1x = walker + t.dot(walker)
    # olp = trial._calc_overlap_restricted(walker_1x, wave_data)
    olp = _hf_walker_olp(walker_1x)
    return olp

@jax.jit
def _hf_olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array) -> complex:
    '''
    <psi_0|exp(x*h2_mod)|walker>
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _hf_walker_olp(walker_2x)
    # olp = trial._calc_overlap_restricted(walker_2x, wave_data)
    return olp

@partial(jit, static_argnums=2)
def _e0(walker, ham_data, trial, eps=3e-4):
    norb = trial.norb
    h0 = ham_data['h0']
    h1_mod = ham_data['h1_mod']
    chol = ham_data["chol"].reshape(-1, norb, norb)

    # one body
    x = 0.0
    f1 = lambda a: _hf_olp_exp1(a,h1_mod,walker)
    olp, d_olp = jvp(f1, [x], [1.0])

    # two body
    def scanned_fun(carry, c):
        eps, walker = carry
        return carry, _hf_olp_exp2(eps,c,walker)

    _, olp_p = lax.scan(scanned_fun, (eps, walker), chol)
    _, olp_0 = lax.scan(scanned_fun, (0.0, walker), chol)
    _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker), chol)
    d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps

    e_0= h0 + (d_olp + jnp.sum(d_2_olp) / 2.0 )/ olp

    return e_0

### e1 part ###
@jax.jit
def _ci1_walker_olp(walker: jax.Array,wave_data: dict):
    '''ci_ia <psi_i^a|phi>'''
    ci1= wave_data['ci1']
    nocc = walker.shape[1]
    GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
    return 2 * o0 * o1

@jax.jit
def _ci1_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    c_ia <psi_i^a|exp(x*h1_mod)|walker>
    '''
    t = x * h1_mod
    walker_1x = walker + t.dot(walker)
    olp = _ci1_walker_olp(walker_1x,wave_data)
    return olp

@jax.jit
def _ci1_olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    c_ia <psi_i^a|exp(x*h2_mod)|walker>
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _ci1_walker_olp(walker_2x,wave_data)
    return olp

@partial(jit, static_argnums=3)
def _e1(walker,ham_data,wave_data,trial,eps=3e-4):
    norb = trial.norb
    h0 = ham_data['h0']
    h1_mod = ham_data['h1_mod']
    chol = ham_data["chol"].reshape(-1, norb, norb)

    hf_olp = _hf_walker_olp(walker)
    ci1_olp = _ci1_walker_olp(walker,wave_data)

    # one body
    x = 0.0
    f1 = lambda a: _ci1_olp_exp1(a,h1_mod,walker,wave_data)
    _, d_olp = jvp(f1, [x], [1.0])

    # two body
    def scanned_fun(carry, c):
        eps, walker, wave_data = carry
        return carry, _ci1_olp_exp2(eps,c,walker,wave_data)

    _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps

    e_1 = (h0 * ci1_olp + d_olp + jnp.sum(d_2_olp) / 2.0 ) / hf_olp

    return e_1


### e2 part ###
@jax.jit
def _ci2_walker_olp(walker,wave_data):
    ''' c_ij^ab <psi_ij^ab|phi> '''
    ci2 = wave_data['ci2']
    nocc = walker.shape[1]
    GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    o2 = 2 * jnp.einsum(
        "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]
    ) - jnp.einsum("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:])
    return o2 * o0

@jax.jit
def _ci2_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    c_ia <psi_i^a|exp(x*h1_mod)|walker>
    '''
    t = x * h1_mod
    walker_1x = walker + t.dot(walker)
    olp = _ci2_walker_olp(walker_1x,wave_data)
    return olp

@jax.jit
def _ci2_olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    c_ia <psi_i^a|exp(x*h2_mod)|walker>
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _ci2_walker_olp(walker_2x,wave_data)
    return olp

@partial(jit, static_argnums=3)
def _e2(walker, ham_data, wave_data, trial, eps=3e-4):
    norb = trial.norb
    h0 = ham_data['h0']
    h1_mod = ham_data['h1_mod']
    chol = ham_data["chol"].reshape(-1, norb, norb)

   
    # zero body
    # c_ij^ab <psi_ij^ab|phi>/<psi_0|phi> * h0
    hf_olp = _hf_walker_olp(walker)
    ci2_olp = _ci2_walker_olp(walker,wave_data)

    # one body
    # c_ij^ab <psi_ij^ab|phi_1x>/<psi_0|phi>
    x = 0.0
    f1 = lambda a: _ci2_olp_exp1(a,h1_mod,walker,wave_data)
    _, d_olp = jvp(f1, [x], [1.0])

    # two body
    # c_ij^ab <psi_ij^ab|phi_2x>/<psi_0|phi>
    def scanned_fun(carry, c):
        eps, walker, wave_data = carry
        return carry, _ci2_olp_exp2(eps,c,walker,wave_data)

    _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps
    
   
    e_2 = (h0 * ci2_olp + d_olp + jnp.sum(d_2_olp) / 2.0 ) / hf_olp

    return e_2

@partial(jit, static_argnums=3)
def _cisd_walker_energy_pt(walker,ham_data,wave_data,trial):
    ci1,ci2 = wave_data['ci1'],wave_data['ci2']
    nocc = trial.nelec[0]
    green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
    green = green[:nocc,:]
    green_occ = green[:, nocc:].copy()
    ci1g = jnp.einsum("ia,ia->", ci1, green_occ, optimize="optimal")
    ci2g_c = jnp.einsum("iajb,ia->jb", ci2, green_occ)
    ci2g_e = jnp.einsum("iajb,ib->ja", ci2, green_occ)
    ci2g = 2 * ci2g_c - ci2g_e
    gci2g = jnp.einsum("ia,ia->", ci2g, green_occ, optimize="optimal")
    # olp = 1 + 2*ci1g + gci2g
    c1 = 2*ci1g
    c2 = gci2g
    e0 = _e0(walker,ham_data,trial)
    e1 = _e1(walker,ham_data,wave_data,trial)
    e2 = _e2(walker,ham_data,wave_data,trial)
    E0 = e0
    E1 = e1 - c1*e0
    E2 = e2 - c1*e1 + (c1**2-c2)*e0
    return jnp.real(E0+E1+E2)

@partial(jit, static_argnums=3)
def cisd_walker_energy_pt(walkers,ham_data,wave_data,trial):
    e_pt= vmap(
        _cisd_walker_energy_pt,in_axes=(0, None, None, None))(
        walkers, ham_data, wave_data,trial)
    return e_pt 