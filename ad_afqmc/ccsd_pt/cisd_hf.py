import jax
from jax import lax, jit, jvp, vmap
from jax import numpy as jnp
from functools import partial

### e0 part ###
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
def _ccsd_walker_energy_pt(walker, ham_data, wave_data, trial):
    ''' <psi_0|(t1+t2)(h1+h2)|phi>/<psi_0|phi> '''

    eps=3e-4

    norb = trial.norb
    h1_mod = ham_data['h1_mod']
    chol = ham_data["chol"].reshape(-1, norb, norb)

    # one body
    x = 0.0
    f1 = lambda a: _t1t2_olp_exp1(a,h1_mod,walker,wave_data)
    olp_t, d_olp = jvp(f1, [x], [1.0])

    # two body
    # c_ij^ab <psi_ij^ab|phi_2x>/<psi_0|phi>
    def scanned_fun(carry, c):
        eps, walker, wave_data = carry
        return carry, _t1t2_olp_exp2(eps,c,walker,wave_data)

    _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
    _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
    _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
    d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps
    
    o0 = trial._calc_overlap_restricted(walker, wave_data)
    t = olp_t/o0
    e0 = trial._calc_energy_restricted(walker,ham_data,wave_data)
    e1 = (d_olp + jnp.sum(d_2_olp) / 2.0 ) / o0

    return jnp.real(t), jnp.real(e0), jnp.real(e1)

@partial(jit, static_argnums=3) 
def ccsd_walker_energy_pt(walkers,ham_data,wave_data,trial):
    t, e0, e1 = vmap(
        _ccsd_walker_energy_pt,in_axes=(0, None, None, None))(
        walkers, ham_data, wave_data,trial)
    return t, e0, e1

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
    h0 = ham_data["h0"]
    t = overlap_1 + overlap_2 # <psi|(t1+t2)|phi>/<psi|phi>
    e_0 = h0 + e1_0 + e2_0 # <psi|h1+h2|phi>/<psi|phi>
    e_12 = e1_1 + e1_2 + e2_1 + e2_2 # <psi|(t1+t2)(h1+h2)|phi>/<psi|phi>

    return jnp.real(t), jnp.real(e_0), jnp.real(e_12)

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
        elif isinstance(cc, CCSD):
            ci1 = np.array(cc.t1)
            ci2 = cc.t2 # + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
            ci2 = ci2.transpose(0, 2, 1, 3)
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

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import uccsd


def update_amps(cc, t1, t2, eris):
    # cc = uccsd.UCCSD(mf)
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(cc.stdout, cc.verbose)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + cc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + cc.level_shift
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]

    u1a = np.zeros_like(t1a)
    u1b = np.zeros_like(t1b)
    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:u2aa += lib.einsum('ijef,aebf->ijab', tauaa, eris_vvvv) * .5
    #:u2bb += lib.einsum('ijef,aebf->ijab', taubb, eris_VVVV) * .5
    #:u2ab += lib.einsum('iJeF,aeBF->iJaB', tauab, eris_vvVV)
    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    u2aa, u2ab, u2bb = cc._add_vvvv(None, (tauaa,tauab,taubb), eris)
    u2aa *= .5
    u2bb *= .5

    Fooa =  .5 * lib.einsum('me,ie->mi', fova, t1a)
    Foob =  .5 * lib.einsum('me,ie->mi', fovb, t1b)
    Fvva = -.5 * lib.einsum('me,ma->ae', fova, t1a)
    Fvvb = -.5 * lib.einsum('me,ma->ae', fovb, t1b)
    Fooa += eris.focka[:nocca,:nocca] - np.diag(mo_ea_o)
    Foob += eris.fockb[:noccb,:noccb] - np.diag(mo_eb_o)
    Fvva += eris.focka[nocca:,nocca:] - np.diag(mo_ea_v)
    Fvvb += eris.fockb[noccb:,noccb:] - np.diag(mo_eb_v)
    dtype = u2aa.dtype
    wovvo = np.zeros((nocca,nvira,nvira,nocca), dtype=dtype)
    wOVVO = np.zeros((noccb,nvirb,nvirb,noccb), dtype=dtype)
    woVvO = np.zeros((nocca,nvirb,nvira,noccb), dtype=dtype)
    woVVo = np.zeros((nocca,nvirb,nvirb,nocca), dtype=dtype)
    wOvVo = np.zeros((noccb,nvira,nvirb,nocca), dtype=dtype)
    wOvvO = np.zeros((noccb,nvira,nvira,noccb), dtype=dtype)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now - u2aa.size*8e-6)
    if nvira > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)
            wovvo[p0:p1] += lib.einsum('jf,mebf->mbej', t1a, ovvv)
            u1a += 0.5*lib.einsum('mief,meaf->ia', t2aa[p0:p1], ovvv)
            u2aa[:,p0:p1] += lib.einsum('ie,mbea->imab', t1a, ovvv.conj())
            tmp1aa = lib.einsum('ijef,mebf->ijmb', tauaa, ovvv)
            u2aa -= lib.einsum('ijmb,ma->ijab', tmp1aa, t1a[p0:p1]*.5)
            ovvv = tmp1aa = None

    if nvirb > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)
            wOVVO[p0:p1] = lib.einsum('jf,mebf->mbej', t1b, OVVV)
            u1b += 0.5*lib.einsum('MIEF,MEAF->IA', t2bb[p0:p1], OVVV)
            u2bb[:,p0:p1] += lib.einsum('ie,mbea->imab', t1b, OVVV.conj())
            tmp1bb = lib.einsum('ijef,mebf->ijmb', taubb, OVVV)
            u2bb -= lib.einsum('ijmb,ma->ijab', tmp1bb, t1b[p0:p1]*.5)
            OVVV = tmp1bb = None

    if nvirb > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)
            woVvO[p0:p1] = lib.einsum('JF,meBF->mBeJ', t1b, ovVV)
            woVVo[p0:p1] = lib.einsum('jf,mfBE->mBEj',-t1a, ovVV)
            u1b += lib.einsum('mIeF,meAF->IA', t2ab[p0:p1], ovVV)
            u2ab[p0:p1] += lib.einsum('IE,maEB->mIaB', t1b, ovVV.conj())
            tmp1ab = lib.einsum('iJeF,meBF->iJmB', tauab, ovVV)
            u2ab -= lib.einsum('iJmB,ma->iJaB', tmp1ab, t1a[p0:p1])
            ovVV = tmp1ab = None

    if nvira > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)
            wOvVo[p0:p1] = lib.einsum('jf,MEbf->MbEj', t1a, OVvv)
            wOvvO[p0:p1] = lib.einsum('JF,MFbe->MbeJ',-t1b, OVvv)
            u1a += lib.einsum('iMfE,MEaf->ia', t2ab[:,p0:p1], OVvv)
            u2ab[:,p0:p1] += lib.einsum('ie,MBea->iMaB', t1a, OVvv.conj())
            tmp1abba = lib.einsum('iJeF,MFbe->iJbM', tauab, OVvv)
            u2ab -= lib.einsum('iJbM,MA->iJbA', tmp1abba, t1b[p0:p1])
            OVvv = tmp1abba = None

    eris_ovov = np.asarray(eris.ovov)
    eris_ovoo = np.asarray(eris.ovoo)
    Woooo = lib.einsum('je,nemi->mnij', t1a, eris_ovoo)
    Woooo = Woooo - Woooo.transpose(0,1,3,2)
    Woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
    Woooo += lib.einsum('ijef,menf->mnij', tauaa, eris_ovov) * .5
    u2aa += lib.einsum('mnab,mnij->ijab', tauaa, Woooo*.5)
    Woooo = tauaa = None
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    Fooa += np.einsum('ne,nemi->mi', t1a, ovoo)
    u1a += 0.5*lib.einsum('mnae,meni->ia', t2aa, ovoo)
    wovvo += lib.einsum('nb,nemj->mbej', t1a, ovoo)
    ovoo = eris_ovoo = None

    tilaa = uccsd.make_tau_aa(t2[0], t1a, t1a, fac=0.5)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    Fvva -= .5 * lib.einsum('mnaf,menf->ae', tilaa, ovov)
    Fooa += .5 * lib.einsum('inef,menf->mi', tilaa, ovov)
    Fova = np.einsum('nf,menf->me',t1a, ovov)
    u2aa += ovov.conj().transpose(0,2,1,3) * .5
    wovvo -= 0.5*lib.einsum('jnfb,menf->mbej', t2aa, ovov)
    woVvO += 0.5*lib.einsum('nJfB,menf->mBeJ', t2ab, ovov)
    tmpaa = lib.einsum('jf,menf->mnej', t1a, ovov)
    wovvo -= lib.einsum('nb,mnej->mbej', t1a, tmpaa)
    eris_ovov = ovov = tmpaa = tilaa = None

    eris_OVOV = np.asarray(eris.OVOV)
    eris_OVOO = np.asarray(eris.OVOO)
    WOOOO = lib.einsum('je,nemi->mnij', t1b, eris_OVOO)
    WOOOO = WOOOO - WOOOO.transpose(0,1,3,2)
    WOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
    WOOOO += lib.einsum('ijef,menf->mnij', taubb, eris_OVOV) * .5
    u2bb += lib.einsum('mnab,mnij->ijab', taubb, WOOOO*.5)
    WOOOO = taubb = None
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    Foob += np.einsum('ne,nemi->mi', t1b, OVOO)
    u1b += 0.5*lib.einsum('mnae,meni->ia', t2bb, OVOO)
    wOVVO += lib.einsum('nb,nemj->mbej', t1b, OVOO)
    OVOO = eris_OVOO = None

    tilbb = uccsd.make_tau_aa(t2[2], t1b, t1b, fac=0.5)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Fvvb -= .5 * lib.einsum('MNAF,MENF->AE', tilbb, OVOV)
    Foob += .5 * lib.einsum('inef,menf->mi', tilbb, OVOV)
    Fovb = np.einsum('nf,menf->me',t1b, OVOV)
    u2bb += OVOV.conj().transpose(0,2,1,3) * .5
    wOVVO -= 0.5*lib.einsum('jnfb,menf->mbej', t2bb, OVOV)
    wOvVo += 0.5*lib.einsum('jNbF,MENF->MbEj', t2ab, OVOV)
    tmpbb = lib.einsum('jf,menf->mnej', t1b, OVOV)
    wOVVO -= lib.einsum('nb,mnej->mbej', t1b, tmpbb)
    eris_OVOV = OVOV = tmpbb = tilbb = None

    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    Fooa += np.einsum('NE,NEmi->mi', t1b, eris_OVoo)
    u1a -= lib.einsum('nMaE,MEni->ia', t2ab, eris_OVoo)
    wOvVo -= lib.einsum('nb,MEnj->MbEj', t1a, eris_OVoo)
    woVVo += lib.einsum('NB,NEmj->mBEj', t1b, eris_OVoo)
    Foob += np.einsum('ne,neMI->MI', t1a, eris_ovOO)
    u1b -= lib.einsum('mNeA,meNI->IA', t2ab, eris_ovOO)
    woVvO -= lib.einsum('NB,meNJ->mBeJ', t1b, eris_ovOO)
    wOvvO += lib.einsum('nb,neMJ->MbeJ', t1a, eris_ovOO)
    WoOoO = lib.einsum('JE,NEmi->mNiJ', t1b, eris_OVoo)
    WoOoO+= lib.einsum('je,neMI->nMjI', t1a, eris_ovOO)
    WoOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)
    eris_OVoo = eris_ovOO = None

    eris_ovOV = np.asarray(eris.ovOV)
    WoOoO += lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)
    u2ab += lib.einsum('mNaB,mNiJ->iJaB', tauab, WoOoO)
    WoOoO = None

    tilab = uccsd.make_tau_ab(t2[1], t1 , t1 , fac=0.5)
    Fvva -= lib.einsum('mNaF,meNF->ae', tilab, eris_ovOV)
    Fvvb -= lib.einsum('nMfA,nfME->AE', tilab, eris_ovOV)
    Fooa += lib.einsum('iNeF,meNF->mi', tilab, eris_ovOV)
    Foob += lib.einsum('nIfE,nfME->MI', tilab, eris_ovOV)
    Fova += np.einsum('NF,meNF->me',t1b, eris_ovOV)
    Fovb += np.einsum('nf,nfME->ME',t1a, eris_ovOV)
    u2ab += eris_ovOV.conj().transpose(0,2,1,3)
    wovvo += 0.5*lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
    wOVVO += 0.5*lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
    wOvVo -= 0.5*lib.einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
    woVvO -= 0.5*lib.einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
    woVVo += 0.5*lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
    wOvvO += 0.5*lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
    tmpabab = lib.einsum('JF,meNF->mNeJ', t1b, eris_ovOV)
    tmpbaba = lib.einsum('jf,nfME->MnEj', t1a, eris_ovOV)
    woVvO -= lib.einsum('NB,mNeJ->mBeJ', t1b, tmpabab)
    wOvVo -= lib.einsum('nb,MnEj->MbEj', t1a, tmpbaba)
    woVVo += lib.einsum('NB,NmEj->mBEj', t1b, tmpbaba)
    wOvvO += lib.einsum('nb,nMeJ->MbeJ', t1a, tmpabab)
    tmpabab = tmpbaba = tilab = None

    Fova += fova
    Fovb += fovb
    u1a += fova.conj()
    u1a += np.einsum('ie,ae->ia', t1a, Fvva)
    u1a -= np.einsum('ma,mi->ia', t1a, Fooa)
    u1a -= np.einsum('imea,me->ia', t2aa, Fova)
    u1a += np.einsum('iMaE,ME->ia', t2ab, Fovb)
    u1b += fovb.conj()
    u1b += np.einsum('ie,ae->ia',t1b,Fvvb)
    u1b -= np.einsum('ma,mi->ia',t1b,Foob)
    u1b -= np.einsum('imea,me->ia', t2bb, Fovb)
    u1b += np.einsum('mIeA,me->IA', t2ab, Fova)

    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    wovvo -= eris_oovv.transpose(0,2,3,1)
    wovvo += eris_ovvo.transpose(0,2,1,3)
    oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
    u1a-= np.einsum('nf,niaf->ia', t1a,      oovv)
    tmp1aa = lib.einsum('ie,mjbe->mbij', t1a,      oovv)
    u2aa += 2*lib.einsum('ma,mbij->ijab', t1a, tmp1aa)
    eris_ovvo = eris_oovv = oovv = tmp1aa = None

    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    wOVVO -= eris_OOVV.transpose(0,2,3,1)
    wOVVO += eris_OVVO.transpose(0,2,1,3)
    OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
    u1b-= np.einsum('nf,niaf->ia', t1b,      OOVV)
    tmp1bb = lib.einsum('ie,mjbe->mbij', t1b,      OOVV)
    u2bb += 2*lib.einsum('ma,mbij->ijab', t1b, tmp1bb)
    eris_OVVO = eris_OOVV = OOVV = None

    eris_ooVV = np.asarray(eris.ooVV)
    eris_ovVO = np.asarray(eris.ovVO)
    woVVo -= eris_ooVV.transpose(0,2,3,1)
    woVvO += eris_ovVO.transpose(0,2,1,3)
    u1b+= np.einsum('nf,nfAI->IA', t1a, eris_ovVO)
    tmp1ab = lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
    tmp1ab+= lib.einsum('IE,mjBE->mBjI', t1b, eris_ooVV)
    u2ab -= lib.einsum('ma,mBiJ->iJaB', t1a, tmp1ab)
    eris_ooVV = eris_ovVO = tmp1ab = None

    eris_OOvv = np.asarray(eris.OOvv)
    eris_OVvo = np.asarray(eris.OVvo)
    wOvvO -= eris_OOvv.transpose(0,2,3,1)
    wOvVo += eris_OVvo.transpose(0,2,1,3)
    u1a+= np.einsum('NF,NFai->ia', t1b, eris_OVvo)
    tmp1ba = lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
    tmp1ba+= lib.einsum('ie,MJbe->MbJi', t1a, eris_OOvv)
    u2ab -= lib.einsum('MA,MbIj->jIbA', t1b, tmp1ba)
    eris_OOvv = eris_OVvo = tmp1ba = None

    u2aa += 2*lib.einsum('imae,mbej->ijab', t2aa, wovvo)
    u2aa += 2*lib.einsum('iMaE,MbEj->ijab', t2ab, wOvVo)
    u2bb += 2*lib.einsum('imae,mbej->ijab', t2bb, wOVVO)
    u2bb += 2*lib.einsum('mIeA,mBeJ->IJAB', t2ab, woVvO)
    u2ab += lib.einsum('imae,mBeJ->iJaB', t2aa, woVvO)
    u2ab += lib.einsum('iMaE,MBEJ->iJaB', t2ab, wOVVO)
    u2ab += lib.einsum('iMeA,MbeJ->iJbA', t2ab, wOvvO)
    u2ab += lib.einsum('IMAE,MbEj->jIbA', t2bb, wOvVo)
    u2ab += lib.einsum('mIeA,mbej->jIbA', t2ab, wovvo)
    u2ab += lib.einsum('mIaE,mBEj->jIaB', t2ab, woVVo)
    wovvo = wOVVO = woVvO = wOvVo = woVVo = wOvvO = None

    Ftmpa = Fvva - .5*lib.einsum('mb,me->be', t1a, Fova)
    Ftmpb = Fvvb - .5*lib.einsum('mb,me->be', t1b, Fovb)
    u2aa += lib.einsum('ijae,be->ijab', t2aa, Ftmpa)
    u2bb += lib.einsum('ijae,be->ijab', t2bb, Ftmpb)
    u2ab += lib.einsum('iJaE,BE->iJaB', t2ab, Ftmpb)
    u2ab += lib.einsum('iJeA,be->iJbA', t2ab, Ftmpa)
    Ftmpa = Fooa + 0.5*lib.einsum('je,me->mj', t1a, Fova)
    Ftmpb = Foob + 0.5*lib.einsum('je,me->mj', t1b, Fovb)
    u2aa -= lib.einsum('imab,mj->ijab', t2aa, Ftmpa)
    u2bb -= lib.einsum('imab,mj->ijab', t2bb, Ftmpb)
    u2ab -= lib.einsum('iMaB,MJ->iJaB', t2ab, Ftmpb)
    u2ab -= lib.einsum('mIaB,mj->jIaB', t2ab, Ftmpa)

    eris_ovoo = np.asarray(eris.ovoo).conj()
    eris_OVOO = np.asarray(eris.OVOO).conj()
    eris_OVoo = np.asarray(eris.OVoo).conj()
    eris_ovOO = np.asarray(eris.ovOO).conj()
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    u2aa -= lib.einsum('ma,jbim->ijab', t1a, ovoo)
    u2bb -= lib.einsum('ma,jbim->ijab', t1b, OVOO)
    u2ab -= lib.einsum('ma,JBim->iJaB', t1a, eris_OVoo)
    u2ab -= lib.einsum('MA,ibJM->iJbA', t1b, eris_ovOO)
    eris_ovoo = eris_OVoo = eris_OVOO = eris_ovOO = None

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)

    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    u1a /= eia_a
    u1b /= eia_b

    u2aa /= lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    u2ab /= lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    u2bb /= lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

    time0 = log.timer_debug1('update t1 t2', *time0)
    t1new = 0*u1a, 0*u1b
    t2new = u2aa, u2ab, u2bb
    return t1new, t2new