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
    '''<psi_0|h1+h2|phi>/<psi_0|phi>'''
    norb = trial.norb
    # h0 = ham_data['h0']
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

    # e_0= h0 + (d_olp + jnp.sum(d_2_olp) / 2.0 )/ olp
    e_0= (d_olp + jnp.sum(d_2_olp) / 2.0 )/ olp

    return e_0

### e1 part ###
@jax.jit
def _t1_walker_olp(walker: jax.Array,wave_data: dict):
    '''t_ia <psi_i^a|phi>'''
    t1= wave_data['t1']
    nocc = walker.shape[1]
    GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    o1 = jnp.einsum("ia,ia", t1, GF[:, nocc:])
    return 2 * o0 * o1

@jax.jit
def _t1_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    t_ia <psi_i^a|exp(x*h1_mod)|walker>
    '''
    t = x * h1_mod
    walker_1x = walker + t.dot(walker)
    olp = _t1_walker_olp(walker_1x,wave_data)
    return olp

@jax.jit
def _t1_olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    t_ia <psi_i^a|exp(x*h2_mod)|walker>
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _t1_walker_olp(walker_2x,wave_data)
    return olp

@partial(jit, static_argnums=3)
def _e1(walker,ham_data,wave_data,trial,eps=3e-4):
    norb = trial.norb
    h0 = ham_data['h0']
    h1_mod = ham_data['h1_mod']
    chol = ham_data["chol"].reshape(-1, norb, norb)

    hf_olp = _hf_walker_olp(walker)
    t1_olp = _t1_walker_olp(walker,wave_data)

    # one body
    x = 0.0
    f1 = lambda a: _t1_olp_exp1(a,h1_mod,walker,wave_data)
    _, d_olp = jvp(f1, [x], [1.0])

    # two body
    def scanned_fun(carry, c):
        eps, walker, wave_data = carry
        return carry, _t1_olp_exp2(eps,c,walker,wave_data)

    _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps

    e_1 = (h0 * t1_olp + d_olp + jnp.sum(d_2_olp) / 2.0 ) / hf_olp
    # e_1 = (d_olp + jnp.sum(d_2_olp) / 2.0 ) / hf_olp

    return e_1


### e2 part ###
@jax.jit
def _t2_walker_olp(walker,wave_data):
    ''' t_ij^ab <psi_ij^ab|phi> '''
    t2 = wave_data['t2']
    nocc = walker.shape[1]
    GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    o2 = 2 * jnp.einsum(
        "iajb, ia, jb", t2, GF[:, nocc:], GF[:, nocc:]
    ) - jnp.einsum("iajb, ib, ja", t2, GF[:, nocc:], GF[:, nocc:])
    return o2 * o0

@jax.jit
def _t2_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    c_ia <psi_i^a|exp(x*h1_mod)|walker>
    '''
    t = x * h1_mod
    walker_1x = walker + t.dot(walker)
    olp = _t2_walker_olp(walker_1x,wave_data)
    return olp

@jax.jit
def _t2_olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    c_ia <psi_i^a|exp(x*h2_mod)|walker>
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _t2_walker_olp(walker_2x,wave_data)
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
    t2_olp = _t2_walker_olp(walker,wave_data)

    # one body
    # c_ij^ab <psi_ij^ab|phi_1x>/<psi_0|phi>
    x = 0.0
    f1 = lambda a: _t2_olp_exp1(a,h1_mod,walker,wave_data)
    _, d_olp = jvp(f1, [x], [1.0])

    # two body
    # c_ij^ab <psi_ij^ab|phi_2x>/<psi_0|phi>
    def scanned_fun(carry, c):
        eps, walker, wave_data = carry
        return carry, _t2_olp_exp2(eps,c,walker,wave_data)

    _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps
    
   
    e_2 = (h0 * t2_olp + d_olp + jnp.sum(d_2_olp) / 2.0 ) / hf_olp

    return e_2

### <psi_0|TH|phi> part ###
@jax.jit
def _t1t2_walker_olp(walker,wave_data):
    ''' <psi_0(t1+t2)|phi> '''
    t1, t2 = wave_data['t1'], wave_data['t2']
    nocc = walker.shape[1]
    GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
    o1 = jnp.einsum("ia,ia", t1, GF[:, nocc:])
    o2 = 2 * jnp.einsum(
        "iajb, ia, jb", t2, GF[:, nocc:], GF[:, nocc:]
    ) - jnp.einsum("iajb, ib, ja", t2, GF[:, nocc:], GF[:, nocc:])
    return (2*o1 + o2) * o0

@jax.jit
def _t1t2_olp_exp1(x: float, h1_mod: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    t_ia <psi_i^a|exp(x*h1_mod)|walker>
    '''
    t = x * h1_mod
    walker_1x = walker + t.dot(walker)
    olp = _t1t2_walker_olp(walker_1x,wave_data)
    return olp

@jax.jit
def _t1t2_olp_exp2(x: float, chol_i: jax.Array, walker: jax.Array,
                  wave_data: dict) -> complex:
    '''
    t_ia <psi_i^a|exp(x*h2_mod)|walker>
    '''
    walker_2x = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
    olp = _t1t2_walker_olp(walker_2x,wave_data)
    return olp

@partial(jit, static_argnums=3)
def _et1t2(walker, ham_data, wave_data, trial, eps=3e-4):
    ''' <psi_0|(t1+t2)(h1+h2)|phi>/<psi_0|phi> '''
    norb = trial.norb
    # h0 = ham_data['h0']
    h1_mod = ham_data['h1_mod']
    chol = ham_data["chol"].reshape(-1, norb, norb)

    hf_olp = _hf_walker_olp(walker)

    # one body
    x = 0.0
    f1 = lambda a: _t1t2_olp_exp1(a,h1_mod,walker,wave_data)
    _, d_olp = jvp(f1, [x], [1.0])

    # two body
    # c_ij^ab <psi_ij^ab|phi_2x>/<psi_0|phi>
    def scanned_fun(carry, c):
        eps, walker, wave_data = carry
        return carry, _t1t2_olp_exp2(eps,c,walker,wave_data)

    _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps
    
    e_1 = ( d_olp + jnp.sum(d_2_olp) / 2.0 ) / hf_olp

    return e_1


@partial(jit, static_argnums=3)
def _ccsd_walker_energy_pt(walker,ham_data,wave_data,trial):
    nocc = trial.nelec[0]
    green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
    green = green[:nocc,:]
    green_occ = green[:, nocc:].copy()
    # expand t1 t2 do uncomment below ###############################
    t1 = wave_data['t1']
    t2 = wave_data['t2']
    t1g = jnp.einsum("ia,ia->", t1, green_occ, optimize="optimal")
    t2g_c = jnp.einsum("iajb,ia->jb", t2, green_occ)
    t2g_e = jnp.einsum("iajb,ib->ja", t2, green_occ)
    t2g = 2 * t2g_c - t2g_e
    gt2g = jnp.einsum("ia,ia->", t2g, green_occ, optimize="optimal")
    t = 2 * t1g + gt2g
    ##################################################################
    e0 = _e0(walker,ham_data,trial)
    e1 = _et1t2(walker,ham_data,wave_data,trial)
    # e1 = _e1(walker,ham_data,wave_data,trial)+_e2(walker,ham_data,wave_data,trial)
    # e2 = _e2(walker,ham_data,wave_data,trial)
    # E0 =  h0 + e0
    # E1 = -t*e0 + e1
    # E2 =  c**2*e0 - c*e1
    # E3 = -c**3*e0 + c**2*e1
    # E4 =  c**4*e0 - c**3*e1
    # e_pt = jnp.real(E0+E1)
    # e_og = jnp.real(h0+(e0+e1)/(1+t))
    return jnp.real(e0), jnp.real(e1), jnp.real(t)

@partial(jit, static_argnums=3)
def ccsd_walker_energy_pt(walkers,ham_data,wave_data,trial):
    e0, e1, t = vmap(
        _ccsd_walker_energy_pt,in_axes=(0, None, None, None))(
        walkers, ham_data, wave_data,trial)
    return e0, e1, t

# with uccsd trial
# doesn't support frozen core currently!!!
@partial(jit, static_argnums=4)
def _uccsd_walker_energy_pt(walker_up,walker_dn,ham_data,wave_data,trial):
    # ci coefficient is actual cc amplitude!!!
    nocc_a, ci1_a, ci2_aa = trial.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
    nocc_b, ci1_b, ci2_bb = trial.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
    ci2_ab = wave_data["ci2AB"]
    walker_dn_b = wave_data["mo_coeff"][1].T.dot(walker_dn[:, :nocc_b])
    green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
    green_b = (walker_dn_b.dot(jnp.linalg.inv(walker_dn_b[:nocc_b, :]))).T
    green_occ_a = green_a[:, nocc_a:].copy()
    green_occ_b = green_b[:, nocc_b:].copy()
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(trial.norb - nocc_a)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(trial.norb - nocc_b)))

    chol_a = ham_data["chol"].reshape(-1, trial.norb, trial.norb)
    chol_b = ham_data["chol_b"].reshape(-1, trial.norb, trial.norb)
    rot_chol_a = chol_a[:, :nocc_a, :]
    rot_chol_b = chol_b[:, :nocc_b, :]
    h1_a = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
    h1_b = ham_data["h1_b"]
    hg_a = jnp.einsum("pj,pj->", h1_a[:nocc_a, :], green_a)
    hg_b = jnp.einsum("pj,pj->", h1_b[:nocc_b, :], green_b)
    hg = hg_a + hg_b

    # 0 body energy
    # h0 = ham_data["h0"]

    # 1 body energy
    # ref
    e1_0 = hg

    # single excitations
    ci1g_a = jnp.einsum("pt,pt->", ci1_a, green_occ_a, optimize="optimal")
    ci1g_b = jnp.einsum("pt,pt->", ci1_b, green_occ_b, optimize="optimal")
    ci1g = ci1g_a + ci1g_b
    e1_1_1 = ci1g * hg
    gpci1_a = greenp_a @ ci1_a.T
    gpci1_b = greenp_b @ ci1_b.T
    ci1_green_a = gpci1_a @ green_a
    ci1_green_b = gpci1_b @ green_b
    e1_1_2 = -(
        jnp.einsum("ij,ij->", h1_a, ci1_green_a, optimize="optimal")
        + jnp.einsum("ij,ij->", h1_b, ci1_green_b, optimize="optimal")
    )
    e1_1 = e1_1_1 + e1_1_2

    # double excitations
    ci2g_a = jnp.einsum("ptqu,pt->qu", ci2_aa, green_occ_a) / 4
    ci2g_b = jnp.einsum("ptqu,pt->qu", ci2_bb, green_occ_b) / 4
    ci2g_ab_a = jnp.einsum("ptqu,qu->pt", ci2_ab, green_occ_b)
    ci2g_ab_b = jnp.einsum("ptqu,pt->qu", ci2_ab, green_occ_a)
    gci2g_a = jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
    gci2g_b = jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
    gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
    e1_2_1 = hg * gci2g
    ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
    ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
    ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
    ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
    e1_2_2_a = -jnp.einsum(
        "ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a, optimize="optimal"
    )
    e1_2_2_b = -jnp.einsum(
        "ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b, optimize="optimal"
    )
    e1_2_2 = e1_2_2_a + e1_2_2_b
    e1_2 = e1_2_1 + e1_2_2

    # e1 = e1_0 + e1_1 + e1_2

    # two body energy
    # ref
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
    lg1_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
    lg1_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
    e2_0_2 = (
        -(
            jnp.sum(vmap(lambda x: x * x.T)(lg1_a))
            + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))
        )
        / 2.0
    )
    e2_0 = e2_0_1 + e2_0_2

    # single excitations
    e2_1_1 = e2_0 * ci1g
    lci1g_a = jnp.einsum("gij,ij->g", chol_a, ci1_green_a, optimize="optimal")
    lci1g_b = jnp.einsum("gij,ij->g", chol_b, ci1_green_b, optimize="optimal")
    e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
    ci1g1_a = ci1_a @ green_occ_a.T
    ci1g1_b = ci1_b @ green_occ_b.T
    e2_1_3_1 = jnp.einsum(
        "gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, optimize="optimal"
    ) + jnp.einsum("gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, optimize="optimal")
    lci1g_a = jnp.einsum(
        "gip,qi->gpq", ham_data["lci1_a"], green_a, optimize="optimal"
    )
    lci1g_b = jnp.einsum(
        "gip,qi->gpq", ham_data["lci1_b"], green_b, optimize="optimal"
    )
    e2_1_3_2 = -jnp.einsum(
        "gpq,gqp->", lci1g_a, lg1_a, optimize="optimal"
    ) - jnp.einsum("gpq,gqp->", lci1g_b, lg1_b, optimize="optimal")
    e2_1_3 = e2_1_3_1 + e2_1_3_2
    e2_1 = e2_1_1 + e2_1_2 + e2_1_3

    # double excitations
    e2_2_1 = e2_0 * gci2g
    lci2g_a = jnp.einsum(
        "gij,ij->g",
        chol_a,
        8 * ci2_green_a + 2 * ci2_green_ab_a,
        optimize="optimal",
    )
    lci2g_b = jnp.einsum(
        "gij,ij->g",
        chol_b,
        8 * ci2_green_b + 2 * ci2_green_ab_b,
        optimize="optimal",
    )
    e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

    def scanned_fun(carry, x):
        chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
        gl_a_i = jnp.einsum("pj,ji->pi", green_a, chol_a_i, optimize="optimal")
        gl_b_i = jnp.einsum("pj,ji->pi", green_b, chol_b_i, optimize="optimal")
        lci2_green_a_i = jnp.einsum(
            "pi,ji->pj",
            rot_chol_a_i,
            8 * ci2_green_a + 2 * ci2_green_ab_a,
            optimize="optimal",
        )
        lci2_green_b_i = jnp.einsum(
            "pi,ji->pj",
            rot_chol_b_i,
            8 * ci2_green_b + 2 * ci2_green_ab_b,
            optimize="optimal",
        )
        carry[0] += 0.5 * (
            jnp.einsum("pi,pi->", gl_a_i, lci2_green_a_i, optimize="optimal")
            + jnp.einsum("pi,pi->", gl_b_i, lci2_green_b_i, optimize="optimal")
        )
        glgp_a_i = jnp.einsum(
            "pi,it->pt", gl_a_i, greenp_a, optimize="optimal"
        ).astype(jnp.complex64)
        glgp_b_i = jnp.einsum(
            "pi,it->pt", gl_b_i, greenp_b, optimize="optimal"
        ).astype(jnp.complex64)
        l2ci2_a = 0.5 * jnp.einsum(
            "pt,qu,ptqu->",
            glgp_a_i,
            glgp_a_i,
            ci2_aa.astype(jnp.float32),
            optimize="optimal",
        )
        l2ci2_b = 0.5 * jnp.einsum(
            "pt,qu,ptqu->",
            glgp_b_i,
            glgp_b_i,
            ci2_bb.astype(jnp.float32),
            optimize="optimal",
        )
        l2ci2_ab = jnp.einsum(
            "pt,qu,ptqu->",
            glgp_a_i,
            glgp_b_i,
            ci2_ab.astype(jnp.float32),
            optimize="optimal",
        )
        carry[1] += l2ci2_a + l2ci2_b + l2ci2_ab
        return carry, 0.0

    [e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
    )
    e2_2_2 = e2_2_2_1 + e2_2_2_2
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    # e2 = e2_0 + e2_1 + e2_2

    # overlap
    overlap_1 = ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
    overlap_2 = gci2g
    # overlap = 1.0 + overlap_1 + overlap_2

    # e1 = e1_0 + e1_1 + e1_2
    # e2 = e2_0 + e2_1 + e2_2
    # e_ccsd = (e1 + e2) / overlap + h0

    t = overlap_1 + overlap_2 # <psi|(t1+t2)|phi>/<psi|phi>
    e_0 = e1_0 + e2_0 # <psi|h1+h2|phi>/<psi|phi>
    e_12 = e1_1 + e1_2 + e2_1 + e2_2 # <psi|(t1+t2)(h1+h2)|phi>/<psi|phi>

    return jnp.real(t), jnp.real(e_0), jnp.real(e_12) # doesn't contain h0!!!

@partial(jit, static_argnums=3)
def uccsd_walker_energy_pt(walkers,ham_data,wave_data,trial):
    t, e0, e1 = vmap(
        _uccsd_walker_energy_pt,in_axes=(0, 0, None, None, None))(
        walkers[0], walkers[1], ham_data, wave_data,trial)
    return t, e0, e1

from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf.scf.hf import RHF
from pyscf.scf.uhf import UHF
from pyscf import mcscf
from ad_afqmc import pyscf_interface
import numpy as np

def prep_afqmc(mf_or_cc,basis_coeff=None,
               norb_frozen=0,chol_cut= 1e-5,
               mo_file = "mo_coeff.npz",
               amp_file = "amplitudes.npz",
               chol_file = "FCIDUMP_chol"):

    print("#\n# Preparing AFQMC calculation")

    if isinstance(mf_or_cc, (CCSD, UCCSD)):
        # raise warning about mpi
        print(
            "# If you import pyscf cc modules and use MPI for AFQMC in the same script, finalize MPI before calling the AFQMC driver."
        )
        mf = mf_or_cc._scf
        cc = mf_or_cc
        if cc.frozen is not None:
            norb_frozen = cc.frozen
        if isinstance(cc, UCCSD):
            ci2aa = cc.t2[0] #+ 2 * np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[0])
            ci2aa = (ci2aa - ci2aa.transpose(0, 1, 3, 2)) / 2
            ci2aa = ci2aa.transpose(0, 2, 1, 3)
            ci2bb = cc.t2[2] #+ 2 * np.einsum("ia,jb->ijab", cc.t1[1], cc.t1[1])
            ci2bb = (ci2bb - ci2bb.transpose(0, 1, 3, 2)) / 2
            ci2bb = ci2bb.transpose(0, 2, 1, 3)
            ci2ab = cc.t2[1] #+ np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[1])
            ci2ab = ci2ab.transpose(0, 2, 1, 3)
            ci1a = np.array(cc.t1[0])
            ci1b = np.array(cc.t1[1])
            
            np.savez(
                amp_file,
                ci1a=ci1a,
                ci1b=ci1b,
                ci2aa=ci2aa,
                ci2ab=ci2ab,
                ci2bb=ci2bb,
            )
        else:
            ci2 = cc.t2 # + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
            ci2 = ci2.transpose(0, 2, 1, 3)
            ci1 = np.array(cc.t1)
            np.savez(amp_file, ci1=ci1, ci2=ci2)
    else:
        mf = mf_or_cc

    mol = mf.mol
    # choose the orbital basis
    if basis_coeff is None:
        if isinstance(mf, UHF):
            basis_coeff = mf.mo_coeff[0]
        else:
            basis_coeff = mf.mo_coeff

    # calculate cholesky integrals
    print("# Calculating Cholesky integrals")
    h1e, chol, nelec, enuc, nbasis, nchol = [None] * 6
    DFbas = None
    if getattr(mf, "with_df", None) is not None:
        print('# Decomposing ERI with DF')
        DFbas = mf.with_df.auxmol.basis  # type: ignore
    h1e, chol, nelec, enuc = pyscf_interface.generate_integrals(
        mol, mf.get_hcore(), basis_coeff, chol_cut, DFbas=DFbas
    ) 
    nbasis = h1e.shape[-1]
    nelec = mol.nelec

    if norb_frozen > 0:
        assert norb_frozen * 2 < sum(
            nelec
        ), "Frozen orbitals exceed number of electrons"
        mc = mcscf.CASSCF(
            mf, mol.nao - norb_frozen, mol.nelectron - 2 * norb_frozen
        )
        nelec = mc.nelecas  # type: ignore
        mc.mo_coeff = basis_coeff  # type: ignore
        h1e, enuc = mc.get_h1eff()  # type: ignore
        chol = chol.reshape((-1, nbasis, nbasis))
        chol = chol[:, mc.ncore : mc.ncore + mc.ncas, mc.ncore : mc.ncore + mc.ncas]  # type: ignore

    print("# Finished calculating Cholesky integrals\n#")

    nbasis = h1e.shape[-1]
    print("# Size of the correlation space:")
    print(f"# Number of electrons: {nelec}")
    print(f"# Number of basis functions: {nbasis}")
    print(f"# Number of Cholesky vectors: {chol.shape[0]}\n#")
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write trial mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, UHF):
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        # if isinstance(mf, UHF):
        q, r = np.linalg.qr(
            basis_coeff[:, norb_frozen:]
            .T.dot(overlap)
            .dot(mf.mo_coeff[0][:, norb_frozen:])
        )
        sgn = np.sign(r.diagonal())
        q = np.einsum("ij,j->ij", q, sgn)

        uhfCoeffs[:, :nbasis] = q
        q, r = np.linalg.qr(
            basis_coeff[:, norb_frozen:]
            .T.dot(overlap)
            .dot(mf.mo_coeff[1][:, norb_frozen:])
        )
        sgn = np.sign(r.diagonal())
        q = np.einsum("ij,j->ij", q, sgn)

        uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        np.savez(mo_file, mo_coeff=trial_coeffs)

    elif isinstance(mf, RHF):
        q, _ = np.linalg.qr(
            basis_coeff[:, norb_frozen:]
            .T.dot(overlap)
            .dot(mf.mo_coeff[:, norb_frozen:])
        )
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez(mo_file, mo_coeff=trial_coeffs)
    
    pyscf_interface.write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename=chol_file,
        mo_coeffs=trial_coeffs,
    )