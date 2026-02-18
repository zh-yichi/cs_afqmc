from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit, jvp, lax, vmap
import opt_einsum as oe

from ad_afqmc.prop_unrestricted.wavefunctions_restricted import rhf


class wave_function(ABC):
    """Base class for wave functions. Contains methods for wave function measurements.

    The measurement methods support two types of walker batches:

    1) unrestricted: walkers is a list ([up, down]). up and down are jax.Arrays of shapes
    (nwalkers, norb, nelec[sigma]). In this case the _calc_<property> method is mapped over.

    2) restricted (up and down dets are assumed to be the same): walkers is a jax.Array of shape
    (nwalkers, max(nelec[0], nelec[1])). In this case the _calc_<property>_restricted method is mapped over. By default
    this method is defined to call _calc_<property>. For certain trial states, one can override
    it for computational efficiency.

    A minimal implementation of a wave function should define the _calc_<property> methods for
    property = overlap, force_bias, energy.

    The wave function data is stored in a separate wave_data dictionary. Its structure depends on the
    wave function type and is described in the corresponding class. It may contain "rdm1" which is a
    one-body spin RDM (2, norb, norb). If it is not provided, wave function specific methods are called.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_batch: Number of batches used in scan.
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    # @calc_overlap.register
    def calc_overlap(self, walkers: jax.Array, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            overlap_batch = vmap(self._calc_overlap_restricted, in_axes=(0, None))(
                walker_batch, wave_data
            )
            return carry, overlap_batch

        _, overlaps = lax.scan(
            scanned_fun, None, walkers.reshape(self.n_batch, batch_size, self.norb, -1)
        )
        return overlaps.reshape(n_walkers)

    def calc_force_bias(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            fb_batch = vmap(self._calc_force_bias_restricted, in_axes=(0, None, None))(
                walker_batch, ham_data, wave_data
            )
            return carry, fb_batch

        _, fbs = lax.scan(
            scanned_fun, None, walkers.reshape(self.n_batch, batch_size, self.norb, -1)
        )
        return fbs.reshape(n_walkers, -1)

    def calc_energy(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            energy_batch = vmap(self._calc_energy_restricted, in_axes=(0, None, None))(
                walker_batch, ham_data, wave_data
            )
            return carry, energy_batch

        _, energies = lax.scan(
            scanned_fun,
            None,
            walkers.reshape(self.n_batch, batch_size, self.norb, -1),
        )
        return energies.reshape(n_walkers)

    def get_rdm1(self, wave_data: dict) -> jax.Array:
        """Returns the one-body spin reduced density matrix of the trial.
        Used for calculating mean-field shift and as a default value in cases of large
        deviations in observable samples. If wave_data contains "rdm1" this value is used,
        calls otherwise _calc_rdm1.

        Args:
            wave_data : The trial wave function data.

        Returns:
            rdm1: The one-body spin reduced density matrix (2, norb, norb).
        """
        if "rdm1" in wave_data:
            return jnp.array(wave_data["rdm1"])
        else:
            return self._calc_rdm1(wave_data)

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        """Calculate the one-body spin reduced density matrix. Exact or approximate rdm1
        of the trial state.

        Args:
            wave_data : The trial wave function data.

        Returns:
            rdm1: The one-body spin reduced density matrix (2, norb, norb).
        """
        raise NotImplementedError(
            "One-body spin RDM not found in wave_data and not implemented for this trial."
        )

    def get_init_walkers(
        self, wave_data: dict, n_walkers: int, restricted: bool = False
    ) -> Union[Sequence, jax.Array]:
        """Get the initial walkers. Uses the rdm1 natural orbitals.

        Args:
            wave_data: The trial wave function data.
            n_walkers: The number of walkers.
            restricted: Whether the walkers should be restricted.

        Returns:
            walkers: The initial walkers.
                If restricted, a single jax.Array of shape (nwalkers, norb, nelec[0]).
                If unrestricted, a list of two jax.Arrays each of shape (nwalkers, norb, nelec[sigma]).
        """
        rdm1 = self.get_rdm1(wave_data)
        natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, : self.nelec[0]]
        natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, : self.nelec[1]]
        if restricted:
            if self.nelec[0] == self.nelec[1]:
                det_overlap = np.linalg.det(
                    natorbs_up[:, : self.nelec[0]].T @ natorbs_dn[:, : self.nelec[1]]
                )
                if (
                    np.abs(det_overlap) > 1e-3
                ):  # probably should scale this threshold with number of electrons
                    return jnp.array([natorbs_up + 0.0j] * n_walkers)
                else:
                    overlaps = np.array(
                        [
                            natorbs_up[:, i].T @ natorbs_dn[:, i]
                            for i in range(self.nelec[0])
                        ]
                    )
                    new_vecs = natorbs_up[:, : self.nelec[0]] + np.einsum(
                        "ij,j->ij", natorbs_dn[:, : self.nelec[1]], np.sign(overlaps)
                    )
                    new_vecs = np.linalg.qr(new_vecs)[0]
                    det_overlap = np.linalg.det(
                        new_vecs.T @ natorbs_up[:, : self.nelec[0]]
                    ) * np.linalg.det(new_vecs.T @ natorbs_dn[:, : self.nelec[1]])
                    if np.abs(det_overlap) > 1e-3:
                        return jnp.array([new_vecs + 0.0j] * n_walkers)
                    else:
                        raise ValueError(
                            "Cannot find a set of RHF orbitals with good trial overlap."
                        )
            else:
                # bring the dn orbital projection onto up space to the front
                dn_proj = natorbs_up.T.conj() @ natorbs_dn
                proj_orbs = jnp.linalg.qr(dn_proj, mode="complete")[0]
                orbs = natorbs_up @ proj_orbs
                return jnp.array([orbs + 0.0j] * n_walkers)
        else:
            return [
                jnp.array([natorbs_up + 0.0j] * n_walkers),
                jnp.array([natorbs_dn + 0.0j] * n_walkers),
            ]
        
    def decompose_t2(self, wave_data: dict):
        # adapted from Yann

        nO = self.nelec[0]
        nV = self.norb - nO
        nex = nO * nV

        t2 = wave_data['ci2']

        # assert t2.shape == (nO, nO, nV, nV)

        # T2 = LL^T
        # t2 = jnp.einsum("iajb->aibj", t2)
        assert t2.shape == (nO, nV, nO, nV)
        
        t2 = t2.reshape(nex, nex)
        e_val, e_vec = jnp.linalg.eigh(t2)
        L = e_vec @ jnp.diag(jnp.sqrt(e_val + 0.0j))
        assert jnp.abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

        # Summation on the left
        L = L.T.reshape(nex, nO, nV)
        t2_rec = jnp.einsum('gia,gjb->iajb', L, L)
        assert jnp.abs(wave_data['ci2'] - t2_rec).max() < 1e-12

        # wave_data["T2_L"] = L

        return L, e_val

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


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
    def decompose_t2(self, wave_data: dict):
        # adapted from Yann

        nO = self.nelec[0]
        nV = self.norb - nO
        nex = nO * nV

        t2 = wave_data['t2']

        # assert t2.shape == (nO, nO, nV, nV)

        # T2 = LL^T
        # t2 = jnp.einsum("iajb->aibj", t2)
        assert t2.shape == (nO, nV, nO, nV)
        
        t2 = t2.reshape(nex, nex)
        e_val, e_vec = jnp.linalg.eigh(t2)
        L = e_vec @ jnp.diag(jnp.sqrt(e_val + 0.0j))
        assert jnp.abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

        # Summation on the left
        L = L.T.reshape(nex, nO, nV)
        t2_rec = jnp.einsum('gia,gjb->iajb', L, L)
        assert jnp.abs(wave_data['t2'] - t2_rec).max() < 1e-12

        # wave_data["T2_L"] = L

        return L, e_val

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

        xtau =jnp.einsum("sg,gia->sia", fields, tau) # (nslater,nocc,nvir)

        return xtau

    # sto-CCSD correction part
    @partial(jit, static_argnums=0)
    def xtau_exp_energy_one_walker(
            self,
            walker: jax.Array,
            h0: float,
            h1: jax.Array,
            chol: jax.Array,
            xtau: jax.Array,
            )-> complex:
        # <exp(xtau)HF|H|walker>

        norb = self.norb
        nocc = self.nelec[0]
        one = jnp.eye(norb, dtype=jnp.complex128)
        exp_xtau = one.at[:nocc, nocc:].set(xtau)
        slater = jnp.eye(norb, dtype=jnp.complex128)[:,:nocc]
        slater = exp_xtau.T @ slater

        green = (walker @ (
                jnp.linalg.inv(slater.T.conj() @ walker)
                    ) @ slater.T.conj()
                    ).T
        
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1 = 2 * hg
        lg = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
        e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
        e2_2 = oe.contract('gpq,gqp->',lg,lg, backend="jax")
        e2 = e2_1 - e2_2

        olp = jnp.linalg.det(slater.T.conj() @ walker) ** 2
        
        return (h0 + e1 + e2)*olp, olp

    # sto-CCSD correction part
    @partial(jit, static_argnums=0)
    def xtau_cisd_energy_one_walker(
        self,
        walker: jax.Array,
        h0: float,
        h1: jax.Array,
        chol: jax.Array,
        xtau: jax.Array,
    ) -> complex:
        # <[1 + xtau + (xtau)^2]HF|H|walker>
        
        ci1 = xtau
        ci2 = oe.contract('ia,jb->iajb', xtau, xtau, backend='jax')
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:]
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        rot_chol = chol[:, : self.nelec[0], :]
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        ci1g = oe.contract("pt,pt->", ci1, green_occ, backend="jax")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 * oe.contract("ij,ij->", h1, ci1_green, backend="jax")
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_c = oe.contract("ptqu,pt->qu", ci2, green_occ, backend="jax")
        ci2g_e = oe.contract("ptqu,pu->qt", ci2, green_occ, backend="jax")
        ci2_green_c = (greenp @ ci2g_c.T) @ green
        ci2_green_e = (greenp @ ci2g_e.T) @ green
        ci2_green = 2 * ci2_green_c - ci2_green_e
        ci2g = 2 * ci2g_c - ci2g_e
        gci2g = oe.contract("qu,qu->", ci2g, green_occ, backend="jax")
        e1_2_1 = 2 * hg * gci2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, ci2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2
        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = oe.contract("gpj,qj->gpq", rot_chol, green, backend="jax")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = oe.contract("gij,ij->g", chol, ci1_green, backend="jax")
        e2_1_2 = -2 * (lci1g @ lg)

        ci1g1 = ci1 @ green_occ.T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg1, lg1, ci1g1, backend="jax")
        lci1 = oe.contract(
                "git,pt->gip",
                chol[:, :, self.nelec[0] :],
                ci1,
                backend="jax"
            )
        lci1g = oe.contract("gip,qi->gpq", lci1, green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg

        def scanned_fun(carry, x):
            chol_i, rot_chol_i = x
            gl_i = oe.contract("pj,ji->pi", green, chol_i, backend="jax")
            lci2_green_i = oe.contract(
                "pi,ji->pj", rot_chol_i, ci2_green, backend="jax"
            )
            carry[0] += 0.5 * oe.contract(
                "pi,pi->", gl_i, lci2_green_i, backend="jax"
            )
            glgp_i = oe.contract("pi,it->pt", gl_i, greenp, backend="jax")
            l2ci2_1 = oe.contract(
                "pt,qu,ptqu->",
                glgp_i,
                glgp_i,
                ci2,
                backend="jax"
            )
            l2ci2_2 = oe.contract(
                "pu,qt,ptqu->",
                glgp_i,
                glgp_i,
                ci2,
                backend="jax"
            )
            carry[1] += 2 * l2ci2_1 - l2ci2_2
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol, rot_chol))
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)

        e2_2 = e2_2_1 + e2_2_2 + e2_2_3
        e2 = e2_0 + e2_1 + e2_2

        # overlap
        olp_hf = jnp.linalg.det(walker[:nocc,:nocc]) ** 2
        o1 = 2 * ci1g
        o2 = gci2g
        olp_cisd = (1.0 + o1 + o2) * olp_hf

        return h0*olp_cisd + (e1 + e2)*olp_hf, olp_cisd
    
    @partial(jit, static_argnums=0)
    def xtau_exp_cisd_e_olp_one_walker(
        self,
        walker: jax.Array,
        h0: float,
        h1: jax.Array,
        chol: jax.Array,
        xtau: jax.Array,
        ):

        e_exp, o_exp = self.xtau_exp_energy_one_walker(walker, h0, h1, chol, xtau)
        e_cisd, o_cisd = self.xtau_cisd_energy_one_walker(walker, h0, h1, chol, xtau)

        de = e_exp - e_cisd
        do = o_exp - o_cisd
        
        return (de, do)
    
    @partial(jit, static_argnums=0)
    def get_stoccsd_cr_one_walker(self, walker, ham_data, wave_data):
        def scan_xtau_one_walker(carry, xtau: jax.Array):
            (de, do) = self.xtau_exp_cisd_e_olp_one_walker(
                walker,
                ham_data['h0'],
                (ham_data['h1'][0]+ham_data['h1'][0])/2,
                ham_data['chol'].reshape(-1,self.norb,self.norb),
                xtau,
                )
            return carry, (de, do)

        init_carry = 0.0
        _, (des, dos) = lax.scan(scan_xtau_one_walker, init_carry, wave_data['xtau'])

        de = jnp.sum(des)
        do = jnp.sum(dos)

        return de, do
    
    @partial(jit, static_argnums=(0))
    def calc_stoccsd_cr(self, walkers, ham_data, wave_data):
        de, do = vmap(
            self.get_stoccsd_cr_one_walker,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        
        return de, do

    # @partial(jit, static_argnums=0)
    # def _calc_energy_cisd_hf(
    #     self, walker: jax.Array, ham_data: dict, wave_data: dict
    # ) -> complex:
    #     ci1, ci2 = wave_data["ci1"], wave_data["t2"]
    #     nocc = self.nelec[0]
    #     green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
    #     green_occ = green[:, nocc:].copy()
    #     greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

    #     chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
    #     rot_chol = chol[:, : self.nelec[0], :]
    #     h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
    #     hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

    #     # 0 body energy
    #     h0 = ham_data["h0"]

    #     # 1 body energy
    #     # ref
    #     e1_0 = 2 * hg

    #     # single excitations
    #     ci1g = oe.contract("pt,pt->", ci1, green_occ, backend="jax")
    #     e1_1_1 = 4 * ci1g * hg
    #     gpci1 = greenp @ ci1.T
    #     ci1_green = gpci1 @ green
    #     e1_1_2 = -2 * oe.contract("ij,ij->", h1, ci1_green, backend="jax")
    #     e1_1 = e1_1_1 + e1_1_2

    #     # double excitations
    #     ci2g_c = oe.contract("ptqu,pt->qu", ci2, green_occ, backend="jax")
    #     ci2g_e = oe.contract("ptqu,pu->qt", ci2, green_occ, backend="jax")
    #     ci2_green_c = (greenp @ ci2g_c.T) @ green
    #     ci2_green_e = (greenp @ ci2g_e.T) @ green
    #     ci2_green = 2 * ci2_green_c - ci2_green_e
    #     ci2g = 2 * ci2g_c - ci2g_e
    #     gci2g = oe.contract("qu,qu->", ci2g, green_occ, backend="jax")
    #     e1_2_1 = 2 * hg * gci2g
    #     e1_2_2 = -2 * oe.contract("ij,ij->", h1, ci2_green, backend="jax")
    #     e1_2 = e1_2_1 + e1_2_2
    #     # e1 = e1_0 + e1_1 + e1_2

    #     # two body energy
    #     # ref
    #     lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")
    #     # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
    #     lg1 = oe.contract("gpj,qj->gpq", rot_chol, green, backend="jax")
    #     e2_0_1 = 2 * lg @ lg
    #     e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
    #     e2_0 = e2_0_1 + e2_0_2

    #     # single excitations
    #     e2_1_1 = 2 * e2_0 * ci1g
    #     lci1g = oe.contract("gij,ij->g", chol, ci1_green, backend="jax")
    #     e2_1_2 = -2 * (lci1g @ lg)

    #     ci1g1 = ci1 @ green_occ.T
    #     # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
    #     e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg1, lg1, ci1g1, backend="jax")
    #     lci1g = oe.contract("gip,qi->gpq", ham_data["lci1"], green, backend="jax")
    #     e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
    #     e2_1_3 = e2_1_3_1 + e2_1_3_2
    #     e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

    #     # double excitations
    #     e2_2_1 = e2_0 * gci2g
    #     lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
    #     e2_2_2_1 = -lci2g @ lg

    #     # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
    #     # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
    #     # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
    #     def scanned_fun(carry, x):
    #         chol_i, rot_chol_i = x
    #         gl_i = oe.contract("pj,ji->pi", green, chol_i, backend="jax")
    #         lci2_green_i = oe.contract(
    #             "pi,ji->pj", rot_chol_i, ci2_green, backend="jax"
    #         )
    #         carry[0] += 0.5 * oe.contract(
    #             "pi,pi->", gl_i, lci2_green_i, backend="jax"
    #         )
    #         glgp_i = oe.contract("pi,it->pt", gl_i, greenp, backend="jax")
    #         l2ci2_1 = oe.contract(
    #             "pt,qu,ptqu->",
    #             glgp_i,
    #             glgp_i,
    #             ci2,
    #             backend="jax"
    #         )
    #         l2ci2_2 = oe.contract(
    #             "pu,qt,ptqu->",
    #             glgp_i,
    #             glgp_i,
    #             ci2,
    #             backend="jax"
    #         )
    #         carry[1] += 2 * l2ci2_1 - l2ci2_2
    #         return carry, 0.0

    #     [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol, rot_chol))
    #     e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)

    #     e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    #     olp = 1 + 2*ci1g + gci2g # <CISD|walker>/HF|walker>
    #     e_hf = h0 + e1_0 + e2_0 # <HF|(h1+h2)|phi>/<HF|phi>
    #     # h0 + {<HF|(1+C1+C2)(h1+h2)|walker>/<HF|walker>} / {<CISD|walker>/<HF|walker>}
    #     e_ci = h0 + (e1_0 + e2_0 + e1_1 + e1_2 + e2_1 + e2_2) / olp

    #     return olp, e_hf, e_ci

    @partial(jit, static_argnums=0)
    def _calc_energy_et1_ci(
        self, 
        walker: jax.Array, 
        ham_data: dict, 
        wave_data: dict,
        ci1:  jax.Array,
        ci2:  jax.Array,
        ):
        # calculate <exp(T1)(ci1+ci2)HF|H|walker>

        nocc = self.nelec[0]
        # h0  = ham_data['h0']
        mo_t = wave_data["mo_t"]
        # t2 = wave_data["t2"]
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        green = (walker @ (jnp.linalg.inv(mo_t.T @ walker)) @ mo_t.T).T
        greenp = (green - jnp.eye(self.norb))[:,nocc:]

        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1_0 = 2 * hg

        # one-body
        # single excitations
        ci1g = oe.contract("ia,ia->", ci1, green[:nocc,nocc:], backend="jax")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 * oe.contract("ij,ij->", h1, ci1_green, backend="jax")
        e1_1 = e1_1_1 + e1_1_2 # <exp(T1)HF|ci1 h1|walker> / <exp(T1)HF|walker>

        # double excitations
        t2g_c = oe.contract("iajb,ia->jb", ci2, green[:nocc,nocc:], backend="jax")
        t2g_e = oe.contract("iajb,ib->ja", ci2, green[:nocc,nocc:], backend="jax")
        t2_green_c = (greenp @ t2g_c.T) @ green[:nocc,:]
        t2_green_e = (greenp @ t2g_e.T) @ green[:nocc,:]
        t2_green = 2 * t2_green_c - t2_green_e
        t2g = 2 * t2g_c - t2g_e
        gt2g = oe.contract("ia,ia->", t2g, green[:nocc,nocc:], backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2 # <exp(T1)HF|T2 h1|walker> / <exp(T1)HF|walker>

        # two-body energy
        lg = oe.contract("gpq,pq->g", chol, green, backend="jax")

       # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = oe.contract("gij,ij->g", chol, ci1_green, backend="jax")
        e2_1_2 = -2 * (lci1g @ lg)

        ci1g1 = ci1 @ green[:nocc,nocc:].T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        lci1 = oe.contract(
            "gpa,ia->gpi",
            chol.reshape(-1, self.norb, self.norb)[:, :, nocc:],
            ci1, optimize="optimal", backend="jax"
        )
        lg1 = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
        e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg1, lg1, ci1g1, backend="jax")
        lci1g = oe.contract("gri,pr->gip", lci1, green, backend="jax")
        e2_1_3_2 = -oe.contract("gip,gpi->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3) # <exp(T1)HF|ci1 h2|walker> / <exp(T1)HF|walker>

        # double excitations
        # e2_2_1 = e2_0 * gt2g
        lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        e2_2_2_1 = -lt2g @ lg

        def scanned_fun(carry, x):
            chol_i = x
            # e2_0
            lg_i = oe.contract("pr,qr->pq", chol_i, green, backend="jax")
            e2_0_1_i = (2*jnp.trace(lg_i))**2 / 2.0
            e2_0_2_i = - oe.contract('pq,qp->',lg_i,lg_i, backend="jax")
            carry[0] += e2_0_1_i + e2_0_2_i
            # e2_2
            gl_i = oe.contract("pr,rq->pq",green,chol_i,backend="jax")
            lt2_green_i = oe.contract("pr,qr->pq",chol_i,t2_green,backend="jax")
            carry[1] += 0.5 * oe.contract("pq,pq->",gl_i,lt2_green_i,backend="jax")
            glgp_i = oe.contract("iq,qa->ia", gl_i[:nocc,:],greenp,backend="jax")
            l2t2_1 = oe.contract("ia,jb,iajb->",glgp_i,glgp_i,ci2,backend="jax")
            l2t2_2 = oe.contract("ib,ja,iajb->",glgp_i,glgp_i,ci2,backend="jax")
            carry[2] += 2 * l2t2_1 - l2t2_2
            return carry, 0.0

        [e2_0, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0,0.0,0.0], chol)
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        # o0 = jnp.linalg.det(walker[:nocc,:nocc]) ** 2 # <HF|walker>
        # t1 = jnp.linalg.det(wave_data["mo_t"].T.conj() @ walker)**2 / o0 # <exp(T1)HF|walker>/<HF|walker>
        # t2 = gt2g * t1 # <exp(T1)HF|T2|walker>/<HF|walker>
        # e0 = (e1_0 + e2_0) * t1 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
        # e1 = (e1_2 + e2_2) * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>
        ot1 = jnp.linalg.det(wave_data["mo_t"].T.conj() @ walker)**2 # <exp(T1)HF|walker>
        # e = (e1_0 + e2_0 + e1_2 + e2_2) / (1.0 + gt2g) # (<exp(T1)HF|h1+h2|walker> + <exp(T1)T2HF|(h1+h2)|walker>) / <exp(T1)HF|walker>
        olp = (1.0 + gt2g) * ot1
        e = (e1_0 + e2_0 + e1_1 + e2_1 + e1_2 + e2_2) * ot1 # <exp(T1)HF|h1+h2|walker> + <exp(T1)ci1 HF|(h1+h2)|walker> + <exp(T1)ci2 HF|(h1+h2)|walker>

        return olp, e

    @partial(jit, static_argnums=(0))
    def calc_energy_mixed(self,walkers,ham_data,wave_data):
        olp, ehf, eci = vmap(
            self._calc_energy_cisd_hf,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return olp, ehf, eci

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        norb = self.norb
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )

        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = oe.contract(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, norb, norb), backend="jax"
        )
        ham_data["lci1"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            optimize="optimal", backend="jax"
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


class mixed_wave(wave_function):

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