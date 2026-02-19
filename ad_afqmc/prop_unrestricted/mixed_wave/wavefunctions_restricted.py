# from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap
import opt_einsum as oe

from ad_afqmc.prop_unrestricted.wavefunctions_restricted import rhf


class stoccsd(rhf):
    '''
    Trial = Stochastically sampled CCSD wavefunction
    Guide = RHF
    '''

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    nslater: int = 1000

    @partial(jit, static_argnums=(0))
    def get_stocc(self, wave_data: dict, prop_data: dict):
        nO = self.nelec[0]
        nslater = self.nslater
        t1 = wave_data["t1"]

        # L, e_val = hs_op_yann(self, wave_data)
        # L = L.transpose(0,2,1)
        L, _ = self.decompose_t2(wave_data)

        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                nslater,
                L.shape[0],
            ),
        )

        # e^{t1+x*tau2}
        t1s = jnp.array([t1 + 0.0j] * nslater)
        taus = t1s + jnp.einsum("wg,gia->wia", fields, L)

        # from jax import scipy as jsp
        def _exp_tau(tau, sd):
            # tau_full = jnp.zeros((self.norb, self.norb),dtype=jnp.complex128)
            # for matrix that only have one block nonzero exp(tau_ia) = 1 + tau_ia true
            tau_full = jnp.eye(self.norb,dtype=jnp.complex128)
            exp_tau = tau_full.at[:nO, nO:].set(tau)
            # exp_tau = jsp.linalg.expm(tau_full)
            return exp_tau.T @ sd

        # Initial slater determinants
        init_sd = jnp.array([jnp.eye(self.norb)[:,:nO] + 0.0j] * nslater)
        stocc = vmap(_exp_tau)(taus, init_sd)

        return stocc

    @partial(jit, static_argnums=0)
    def get_green_slater(self, trial_slater: jax.Array, walker: jax.Array) -> jax.Array:
        
        green = (
            walker @ (
                jnp.linalg.inv(trial_slater.T.conj() @ walker)
                    ) @ trial_slater.T.conj()
            ).T
        
        return green

    @partial(jit, static_argnums=0)
    def get_energy_slater(self, slater: jax.Array, walker: jax.Array, ham_data: dict) -> jax.Array:
        norb = self.norb

        h0, chol = ham_data["h0"], ham_data["chol"]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = chol.reshape(-1,norb,norb)

        green = self.get_green_slater(slater, walker)
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1 = 2 * hg
        lg = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
        e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
        e2_2 = oe.contract('gpq,gqp->',lg,lg, backend="jax")
        e2 = e2_1 - e2_2

        return h0 + e1 + e2

    @partial(jit, static_argnums=0)
    def get_overlap_slater(self, slater: jax.Array, walker: jax.Array) -> jax.Array:
        return jnp.linalg.det(slater.T.conj() @ walker) ** 2

    @partial(jit, static_argnums=0)
    def get_energy_slaters_one_walker(
        self, 
        slaters: jax.Array,
        walker: jax.Array,
        ham_data: dict
        ):
        """
        slaters: (N, norb, nocc)
        walker:  (norb, nocc)

        returns: (N,) energies
        """

        def scan_slaters(carry, slater):
            # carry is unused; we keep it for scan API
            energy = self.get_energy_slater(slater, walker, ham_data)
            return carry, energy

        # Initial dummy carry (None not allowed)
        init_carry = 0.0

        _, energies = lax.scan(scan_slaters, init_carry, slaters)

        return energies

    @partial(jit, static_argnums=0)
    def get_overlap_slaters_one_walker(
        self,
        slaters: jax.Array,
        walker: jax.Array,
        ):
        """
        slaters: (N, norb, nocc)
        walker:  (norb, nocc)

        returns: (N,) energies
        """

        def scan_slaters(carry, slater):
            # carry is unused; we keep it for scan API
            overlap = self.get_overlap_slater(slater, walker)
            return carry, overlap

        # Initial dummy carry (None not allowed)
        init_carry = 0.0

        _, overlaps = lax.scan(scan_slaters, init_carry, slaters)

        return overlaps
    
    @partial(jit, static_argnums=0)
    def get_eloc_oloc_stocc(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        slaters = wave_data['stocc']
        energies = self.get_energy_slaters_one_walker(slaters, walker, ham_data)
        overlaps = self.get_overlap_slaters_one_walker(slaters, walker) / slaters.shape[0]
        oloc = jnp.sum(overlaps)
        eloc = jnp.sum(overlaps * energies) / oloc
        return (oloc, eloc) 
    
    @partial(jit, static_argnums=0)
    def calc_energy_mixed(
            self, walkers: jax.Array, ham_data: jax.Array, wave_data: dict
            ):

        (overlaps, energies) =  vmap(
            lambda walker: self.get_eloc_oloc_stocc(walker, ham_data, wave_data
            ))(walkers)
        
        return (overlaps, energies)


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))



@dataclass
class stoccsd2(rhf):
    """
    use CISD Trial and HF Guide 
    abosrb the overlap ratio <Trial|walker>/<Guide/walker> into the weight
    w'(walker)  = weight (for measurements) 
                = weight accumulated by HF importance sampling * <CISD|walker>/<HF|walker>
    E_local(walker) = <CISD|H|walker>/<CISD|walker>
    <E> = {sum_walker w'(walker) * E_local(walker)} / {sum_walker w'(walker)}
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    nslater: int = 100

    # @partial(jit, static_argnums=0)
    def get_xtau(self, wave_data: dict, prop_data: dict):
        nslater = self.nslater

        # t_iajb = tau_gia tau_gjb
        tau, _ = self.decompose_t2(wave_data)

        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                nslater,
                tau.shape[0],
            ),
        )

        xtau = jnp.einsum("sg,gia->sia", fields, tau) # (nslater,nocc,nvir)

        return xtau
    
    @partial(jit, static_argnums=0)
    def _green(self, walker: jax.Array, slater: jax.Array) -> jax.Array:
        '''
        full green's function 
        <psi|a_p^dagger a_q|walker>/<exp(T1)HF|walker>
        '''
        green = (walker @ (
            jnp.linalg.inv(slater.T.conj() @ walker)
            ) @ slater.T.conj()).T
        return green

    @partial(jit, static_argnums=0)
    def _slater_olp(self, walker: jax.Array, slater: jax.Array):
        ''' 
        <psi|walker>
        '''
        olp = jnp.linalg.det(slater.T.conj() @ walker) ** 2
        return olp

    @partial(jit, static_argnums=0)
    def _ci_walker_olp(self, walker: jax.Array, slater: jax.Array, ci1: jax.Array, ci2: jax.Array) -> complex:
        ''' 
        <(1+ci1+ci2)psi|walker>
        = c_ia* <psi|ia|walker> + 1/2 c_iajb* <psi|ijab|walker>
        '''
        ci1 = ci1.conj()
        ci2 = ci2.conj()
        nocc = walker.shape[1]
        green_ov = self._green(walker, slater)[:nocc, nocc:]
        o0 = self._slater_olp(walker, slater)
        o1 = 2 * oe.contract("ia,ia-> ", ci1, green_ov, backend="jax")
        o2 = 2 * oe.contract("iajb,ia,jb->", ci2, green_ov, green_ov, backend="jax") \
            - oe.contract("iajb,ib,ja->", ci2, green_ov, green_ov, backend="jax")
        return (1.0 + o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_energy_cisd(
        self, 
        walker: jax.Array, 
        ham_data: dict, 
        wave_data: dict,
        ci1:  jax.Array,
        ci2:  jax.Array,
        ):

        '''
        A local energy evaluator for <psi(ci1+ci2)HF|H|walker> / <psi(ci1+ci2)|walker>
        all operators and the walkers and psi are in the same basis (normally MO)
        |psi> is not necesarily diagonal
        
        all green's function and the chol and ci coeff are as their original definition
        no half rotation performed
        '''

        ci1 = ci1.conj() # applied to the bra
        ci2 = ci2.conj() # applied to the bra

        nocc, norb = self.nelec[0], self.norb
        h0  = ham_data['h0']
        trial_slater = wave_data["mo_t"]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = ham_data["chol"].reshape(-1, norb, norb)
        green = self._green(walker, trial_slater) # full green
        green_ov = green[:nocc, nocc:]
        greenp = (green - jnp.eye(norb))[:, nocc:]
        
        ##################### ref terms #########################
        # one-body 
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1_0 = 2 * hg

        # two-body 
        gl = oe.contract("pr,gqr->gpq", green, chol, backend="jax")
        trgl = oe.contract('gpp->g', gl, backend="jax")
        e2_0_1 = 2 * jnp.sum(trgl**2)
        e2_0_2 = -oe.contract('gpq,gqp->',gl,gl, backend="jax")
        e2_0 = e2_0_1 + e2_0_2
        ##########################################################

        ######################### ci terms #######################
        # one-body single excitations 
        ci1g = oe.contract("ia,ia->", ci1, green_ov, backend="jax")
        e1_1_1 = 4 * ci1g * hg # c_ia G_ia G_pq h_pq
        gpci1 = oe.contract("pa,ia->pi", greenp, ci1, backend="jax")
        ci1_green = oe.contract("pi,iq->pq", gpci1, green[:nocc,:], backend="jax")
        e1_1_2 = -2 * oe.contract("pq,pq->", h1, ci1_green, backend="jax") # c_ia Gp_pa G_iq h_pq
        e1_1 = e1_1_1 + e1_1_2 # <psi|ci1 h1|walker> / <psi|walker>

        # one-body double excitations
        t2g_c = oe.contract("iajb,ia->jb", ci2, green_ov, backend="jax")
        t2g_e = oe.contract("iajb,ib->ja", ci2, green_ov, backend="jax")
        t2_green_c = (greenp @ t2g_c.T) @ green[:nocc,:]
        t2_green_e = (greenp @ t2g_e.T) @ green[:nocc,:]
        t2_green = 2 * t2_green_c - t2_green_e
        t2g = 2 * t2g_c - t2g_e
        gt2g = oe.contract("ia,ia->", t2g, green_ov, backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2 # <exp(T1)HF|T2 h1|walker> / <exp(T1)HF|walker>

        # two-body single excitations
        e2_1_1 = 2 * e2_0 * ci1g # c_ia G_ia G_pr G_ps L_pr L_ps
        lci1g = oe.contract("gpq,pq->g", chol, ci1_green, backend="jax")
        e2_1_2 = -2 * oe.contract("g,g->",lci1g, trgl, backend="jax") # c_ia Gp_pa G_ir G_qs L_pr L_qs

        lci1 = oe.contract("gpa,ia->gpi", chol[:, :, nocc:], ci1, backend="jax")
        lg1 = oe.contract("gpr,qr->gpq", chol, green[:nocc,:], backend="jax")
        lci1g = oe.contract("gri,pr->gip", lci1, green, backend="jax")
        glgpci1 = jnp.einsum("gpr,ri->gpi", gl, gpci1, optimize="optimal")
        e2_1_3 = jnp.einsum("gpi,gpi->", glgpci1, lg1, optimize="optimal")
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3) # <exp(T1)HF|ci1 h2|walker> / <exp(T1)HF|walker>

        # two-body double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        e2_2_2_1 = -oe.contract("g,g->", lt2g, trgl, backend="jax")

        def scan_aux(carry, x):
            chol_i, gl_i = x
            lt2_green_i = oe.contract("pr,qr->pq",chol_i,t2_green,backend="jax")
            carry[0] += 0.5 * oe.contract("pq,pq->",gl_i,lt2_green_i,backend="jax")
            glgp_i = oe.contract("iq,qa->ia", gl_i[:nocc,:],greenp,backend="jax")
            l2t2_1 = oe.contract("ia,jb,iajb->",glgp_i,glgp_i,ci2,backend="jax")
            l2t2_2 = oe.contract("ib,ja,iajb->",glgp_i,glgp_i,ci2,backend="jax")
            carry[1] += 2 * l2t2_1 - l2t2_2
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(scan_aux, [0.0, 0.0], (chol, gl))

        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        o0 = self._slater_olp(walker, trial_slater)
        overlap = (1.0 + 2*ci1g +  gt2g) * o0 # <(1+c1+c2)psi|walker>
        # <psi|h0+h1+h2|walker> + <ci1 psi|h0+h1+h2|walker> + <ci2 pai|h0+h1+h2|walker>
        energy = h0*overlap + (e1_0 + e2_0 + e1_1 + e2_1 + e1_2 + e2_2) * o0

        return overlap, energy
    
    @partial(jit, static_argnums=0)
    def _calc_energy_slater(self, slater: jax.Array, walker: jax.Array, ham_data: dict) -> jax.Array:
        norb = self.norb

        h0, chol = ham_data["h0"], ham_data["chol"]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = chol.reshape(-1, norb, norb)

        green = self._green(walker, slater)
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1 = 2 * hg
        lg = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
        e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
        e2_2 = -oe.contract('gpq,gqp->',lg,lg, backend="jax")
        e2 = e2_1 + e2_2

        overlap = self._slater_olp(walker, slater)
        energy = (h0 + e1 + e2) * overlap

        return overlap, energy

    @partial(jit, static_argnums=0)
    def _calc_energy_exp_xtau(self, walker: jax.Array, ham_data: dict, wave_data: dict, xtau: jax.Array) -> jax.Array:
        
        slater = self._thouless(wave_data['mo_t'], xtau)
        # overlap = self._slater_olp(walker, slater)
        overlap, energy = self._calc_energy_slater(slater, walker, ham_data)

        return overlap, energy

    @partial(jit, static_argnums=0)
    def _calc_energy_ci_xtau(self, walker: jax.Array, ham_data: dict, wave_data: dict, xtau: jax.Array) -> jax.Array:
        ci1 = xtau
        ci2 = oe.contract('ia,jb->iajb', xtau, xtau, backend='jax')
        overlap, energy = self._calc_energy_cisd(walker, ham_data, wave_data, ci1, ci2)
        # TODO: write a separate ci_energy for disconnected ci2
        return overlap, energy


    @partial(jit, static_argnums=0)
    def _calc_exp_ci_xtau(self, walker, ham_data, wave_data, xtau):
        o_exp, e_exp = self._calc_energy_exp_xtau(walker, ham_data, wave_data, xtau)
        o_ci, e_ci =  self._calc_energy_ci_xtau(walker, ham_data, wave_data, xtau)
        do = o_exp - o_ci
        de = e_exp - e_ci
        return (do, de)

    @partial(jit, static_argnums=0)
    def _calc_exp_ci_xtaus(self, walker, ham_data, wave_data):
        def _scan_xtaus(carry, xtau: jax.Array):
            (do, de) = self._calc_exp_ci_xtau(walker, ham_data, wave_data, xtau)
            return carry, (do, de)

        init_carry = 0.0
        _, (dos, des) = lax.scan(_scan_xtaus, init_carry, wave_data['xtau'])

        do = jnp.sum(dos)
        de = jnp.sum(des)

        return (do, de)

    @partial(jit, static_argnums=(0))
    def calc_energy_cr(self, walkers, ham_data, wave_data):

        doverlap, denergy = vmap(
            self._calc_exp_ci_xtaus,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        
        return doverlap, denergy

    @partial(jit, static_argnums=0)
    def _calc_energy_cid(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ):
        nocc = self.nelec[0]
        mo_t, t2 = wave_data["mo_t"], wave_data["t2"]
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        green = (walker @ (jnp.linalg.inv(mo_t.T @ walker)) @ mo_t.T).T
        greenp = (green - jnp.eye(self.norb))[:,nocc:]

        ################## ref ##################
        h0 = ham_data['h0']
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1_0 = 2 * hg

        gl = oe.contract("pr,gqr->gpq", green, chol, backend="jax")
        trgl = oe.contract('gpp->g', gl, backend="jax")
        e2_0_1 = 2 * jnp.sum(trgl**2)
        e2_0_2 = -oe.contract('gpq,gqp->',gl,gl, backend="jax")
        e2_0 = e2_0_1 + e2_0_2
        ############################################

        # one-body double excitations #############
        t2g_c = oe.contract("iajb,ia->jb", t2, green[:nocc,nocc:], backend="jax")
        t2g_e = oe.contract("iajb,ib->ja", t2, green[:nocc,nocc:], backend="jax")
        t2_green_c = (greenp @ t2g_c.T) @ green[:nocc,:]
        t2_green_e = (greenp @ t2g_e.T) @ green[:nocc,:]
        t2_green = 2 * t2_green_c - t2_green_e
        t2g = 2 * t2g_c - t2g_e
        gt2g = oe.contract("ia,ia->", t2g, green[:nocc,nocc:], backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2 # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

        # two body double excitations #############
        e2_2_1 = e2_0 * gt2g
        lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        e2_2_2_1 = -oe.contract("g,g->", lt2g, trgl, backend="jax")

        def scan_aux(carry, x):
            chol_i, gl_i = x
            lt2_green_i = oe.contract("pr,qr->pq", chol_i, t2_green, backend="jax")
            carry[0] += 0.5 * oe.contract("pq,pq->", gl_i, lt2_green_i, backend="jax")
            glgp_i = oe.contract("iq,qa->ia", gl_i[:nocc,:], greenp, backend="jax")
            l2t2_1 = oe.contract("ia,jb,iajb->", glgp_i, glgp_i, t2, backend="jax")
            l2t2_2 = oe.contract("ib,ja,iajb->", glgp_i, glgp_i, t2, backend="jax")
            carry[1] += 2 * l2t2_1 - l2t2_2
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(scan_aux, [0.0, 0.0], (chol, gl))
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3


        o0 = self._slater_olp(walker, wave_data["mo_t"])
        overlap = (1.0 + gt2g) * o0 # denominator
        
        # numerator  
        energy = h0*overlap + (e1_0 + e2_0 + e1_2 + e2_2)*o0

        return overlap, energy

    @partial(jit, static_argnums=(0))
    def calc_energy_ci(self,walkers,ham_data,wave_data):
        overlap, energy = vmap(
            self._calc_energy_cid,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return overlap, energy

    # @partial(jit, static_argnums=0)
    # def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
    #     # norb = self.norb
    #     ham_data["h1"] = (
    #         ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
    #     )

    #     # ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
    #     #     (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
    #     # )
    #     # ham_data["rot_chol"] = oe.contract(
    #     #     "pi,gij->gpj",
    #     #     wave_data["mo_coeff"].T.conj(),
    #     #     ham_data["chol"].reshape(-1, norb, norb), backend="jax"
    #     # )
    #     # ham_data["lci1"] = oe.contract(
    #     #     "git,pt->gip",
    #     #     ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
    #     #     wave_data["ci1"],
    #     #     optimize="optimal", backend="jax"
    #     # )
    #     return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


class mixed_wave:

    def __init__(self, guide, trial):
        self._guide = guide
        self._trial = trial

    def _calc_rdm1(self, wave_data):
        return self._guide._calc_rdm1(wave_data)

    def _calc_force_bias_restricted(self, walker, ham_data, wave_data):
        return self._guide._calc_force_bias_restricted(walker, ham_data, wave_data)

    def _calc_overlap_restricted(self, walker, wave_data):
        return self._guide._calc_overlap_restricted(walker, wave_data)
    
    def _calc_energy_restricted(self, walker, ham_data, wave_data):
        return self._guide._calc_energy_restricted(walker, ham_data, wave_data)
    
    def _calc_energy_mixed(self, walker, ham_data, wave_data):
        eg = self._guide._calc_energy_restricted(walker, ham_data, wave_data)
        et = self._trial._calc_energy_restricted(walker, ham_data, wave_data)
        og = self._guide._calc_overlap_restricted(walker, wave_data)
        ot = self._trial._calc_overlap_restricted(walker, wave_data)
        otg = ot/og
        return jnp.real(otg), jnp.real(eg), jnp.real(et)
    
    @partial(jit, static_argnums=(0))
    def calc_energy_mixed(self,walkers,ham_data,wave_data):
        otg, eg, et = vmap(
            self._calc_energy_mixed,in_axes=(0, None, None))(
                walkers, ham_data, wave_data)
        return otg, eg, et
    
    def _build_measurement_intermediates(self, ham_data, wave_data):
        ham_data = self._guide._build_measurement_intermediates(ham_data, wave_data)
        ham_data = self._trial._build_measurement_intermediates(ham_data, wave_data)
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))