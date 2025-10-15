import jax
from jax import lax, jit, jvp, vmap
from jax import numpy as jnp
from functools import partial
import numpy as np

### <psi|T2(h1+h2)|phi>/<psi|phi> ###
@partial(jit, static_argnums=2)
def _t2_walker_olp(walker,wave_data,trial):
    ''' t_iajb <psi|ijab|phi> '''
    rot_t2 = wave_data['rot_t2']
    nocc = walker.shape[1]
    # GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    GF = trial._calc_green(walker, wave_data)
    # o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    # <psi|phi>
    o0 = trial._calc_overlap_restricted(walker, wave_data)
    # t_iajb <psi|ijab|phi>/<psi|phi>
    o2 = 2 * jnp.einsum(
        "iajb, ia, jb", rot_t2, GF[:, nocc:], GF[:, nocc:]
    ) - jnp.einsum("iajb, ib, ja", rot_t2, GF[:, nocc:], GF[:, nocc:])
    return o2 * o0

@partial(jit, static_argnums=4)
def _t2_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict,trial) -> complex:
    '''
    t_iajb <psi|ijab exp(x*h1_mod)|walker>
    '''
    t = x * h1_mod
    walker_1x = walker + t.dot(walker)
    olp = _t2_walker_olp(walker_1x,wave_data,trial)
    return olp

@partial(jit, static_argnums=4)
def _t2_olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array,
                  wave_data: dict, trial) -> complex:
    '''
    t_iajb <psi|ijab exp(x*h2_mod)|walker>
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _t2_walker_olp(walker_2x,wave_data,trial)
    return olp

@partial(jit, static_argnums=3)
def _ccsd_walker_energy_pt2(walker, ham_data, wave_data, trial):
    ''' 
    t = <psi|T2|phi>/<psi|phi>
    e0 = <psi|H|phi>/<psi|phi>
    e1 = <psi|T2(h1+h2)|phi>/<psi|phi>
    '''
    eps=3e-4

    norb = trial.norb
    # h0 = ham_data['h0']
    h1_mod = ham_data['h1_mod']
    chol = ham_data["chol"].reshape(-1, norb, norb)

    # one body
    # t_ij^ab <psi|ijab|phi_1x>
    x = 0.0
    f1 = lambda a: _t2_olp_exp1(a,h1_mod,walker,wave_data,trial)
    t_olp, d_olp = jvp(f1, [x], [1.0])

    # two body
    # t_ij^ab <psi|ijab|phi_2x>
    def scanned_fun(carry, c):
        eps, walker, wave_data = carry
        return carry, _t2_olp_exp2(eps,c,walker,wave_data,trial)

    _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps
   
    o0 = trial._calc_overlap_restricted(walker, wave_data)
    et2 = (d_olp + jnp.sum(d_2_olp) / 2.0 ) / o0

    # t2 = _t2_walker_olp(walker,wave_data,trial)/o0
    t2 = t_olp/o0
    e0 = trial._calc_energy_restricted(walker,ham_data,wave_data)

    return jnp.real(t2), jnp.real(e0), jnp.real(et2)

@partial(jit, static_argnums=3)
def ccsd_walker_energy_pt2(walkers,ham_data,wave_data,trial):
    t, e0, e1 = vmap(
        _ccsd_walker_energy_pt2,in_axes=(0, None, None, None))(
        walkers, ham_data, wave_data, trial)
    return t, e0, e1

def thouless_trans(t1):
    q, r = np.linalg.qr(t1)
    u_ai = r.T
    u_ji = q
    u_occ = np.vstack((u_ji,u_ai))
    u, _, _ = np.linalg.svd(u_occ)
    return u
