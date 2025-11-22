from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from typing import Any, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import jit, jvp, lax, vjp, vmap
import opt_einsum as oe

from ad_afqmc import linalg_utils


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

    @singledispatchmethod
    def calc_overlap(self, walkers, wave_data: dict) -> jax.Array:
        """Calculate the overlap < psi_t | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The overlap of the walkers with the trial wave function.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_overlap.register
    def _(self, walkers: list, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            overlap_batch = vmap(self._calc_overlap, in_axes=(0, 0, None))(
                walker_batch_0, walker_batch_1, wave_data
            )
            return carry, overlap_batch

        _, overlaps = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        return overlaps.reshape(n_walkers)

    @calc_overlap.register
    def _(self, walkers: jax.Array, wave_data: dict) -> jax.Array:
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

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Overlap for a single restricted walker."""
        return self._calc_overlap(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], wave_data
        )

    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Overlap for a single walker."""
        raise NotImplementedError("Overlap not defined")

    @singledispatchmethod
    def calc_force_bias(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the force bias < psi_T | chol | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The force bias.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_force_bias.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            fb_batch = vmap(self._calc_force_bias, in_axes=(0, 0, None, None))(
                walker_batch_0, walker_batch_1, ham_data, wave_data
            )
            return carry, fb_batch

        _, fbs = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        fbs = jnp.concatenate(fbs, axis=0)
        return fbs.reshape(n_walkers, -1)

    @calc_force_bias.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
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

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Force bias for a single restricted walker."""
        return self._calc_force_bias(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Force bias for a single walker."""
        raise NotImplementedError("Force bias not definedr")

    @singledispatchmethod
    def calc_energy(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the energy < psi_T | H | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The energy.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_energy.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            energy_batch = vmap(self._calc_energy, in_axes=(0, 0, None, None))(
                walker_batch_0, walker_batch_1, ham_data, wave_data
            )
            return carry, energy_batch

        _, energies = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        return energies.reshape(n_walkers)

    @calc_energy.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
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
    
    @singledispatchmethod
    def calc_orbenergy(self, walkers,ham_data:dict , wave_data:dict, orbE:int) -> jnp.ndarray:
        return vmap(self._calc_orbenergy, in_axes=(None, None, None, 0, None,None))(
            ham_data['h0'], ham_data['rot_h1'], ham_data['rot_chol'], walkers, wave_data,orbE)

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Energy for a single restricted walker."""
        return self._calc_energy(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Energy for a single walker."""
        raise NotImplementedError("Energy not defined")

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

    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Build intermediates for measurements in ham_data. This method is called by the hamiltonian class.

        Args:
            ham_data: The hamiltonian data.
            wave_data: The trial wave function data.

        Returns:
            ham_data: The updated Hamiltonian data.
        """
        return ham_data

    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
        """Optimize the wave function parameters.

        Args:
            ham_data: The hamiltonian data.
            wave_data: The trial wave function data.

        Returns:
            wave_data: The updated trial wave function data.
        """
        return wave_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


class wave_function_cpmc(wave_function):
    """This is used in CPMC. Not as well tested and supported as ab initio currently."""

    @abstractmethod
    def calc_green_diagonal_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        """Calculate the diagonal elements of the greens function.

        Args:
            walkers: The walkers. (mapped over)
            wave_data: The trial wave function data.

        Returns:
            diag_green: The diagonal elements of the greens function.
        """
        pass

    @abstractmethod
    def calc_full_green_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        """Calculate the greens function.

        Args:
            walkers: The walkers. (mapped over)
            wave_data: The trial wave function data.

        Returns:
            green: The greens function.
        """
        pass

    @abstractmethod
    def calc_overlap_ratio_vmap(
        self, greens: Sequence, update_indices: jax.Array, update_constants: jnp.array
    ) -> jax.Array:
        """Calculate the overlap ratio.

        Args:
            greens: The greens functions. (mapped over)
            update_indices: Proposed update indices.
            constants: Proposed update constants.

        Returns:
            overlap_ratios: The overlap ratios.
        """
        pass

    @abstractmethod
    def update_greens_function_vmap(
        self,
        greens: Sequence,
        ratios: jax.Array,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        """Update the greens function.

        Args:
            greens: The old greens functions. (mapped over)
            ratios: The overlap ratios. (mapped over)
            indices: Where to update.
            constants: What to update with. (mapped over)

        Returns:
            green: The updated greens functions.
        """
        pass


# we assume afqmc is performed in the rhf orbital basis
@dataclass
class rhf(wave_function):
    """Class for the restricted Hartree-Fock wave function.

    The corresponding wave_data should contain "mo_coeff", a jax.Array of shape (norb, nelec).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_batch: int = 1

    def __post_init__(self):
        assert (
            self.nelec[0] == self.nelec[1]
        ), "RHF requires equal number of up and down electrons."

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        return jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker) ** 2

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            wave_data["mo_coeff"].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Calculates the half green's function.

        Args:
            walker: The walker.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function.
        """
        return (walker.dot(jnp.linalg.inv(wave_data["mo_coeff"].T.conj() @ walker))).T

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: Sequence, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        green_walker = self._calc_green(walker, wave_data)
        fb = 2.0 * oe.contract("gij,ij->g", ham_data["rot_chol"], green_walker, 
                               backend="jax")
        return fb

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = green_walker_up + green_walker_dn
        fb = oe.contract("gij,ij->g", ham_data["rot_chol"], green_walker, 
                         backend="jax")
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker, wave_data)
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = green_walker_up + green_walker_dn
        ene1 = jnp.sum(green_walker * rot_h1)
        f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = jnp.sum(c * c) - exc
        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

    @partial(jit, static_argnums=0)
    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[0], h1.shape[0]))
        nelec = self.nelec[0]
        h1 = (h1 + h1.T) / 2.0

        def scanned_fun(carry, x):
            dm = carry
            f = oe.contract("gij,ik->gjk", h2, dm, backend="jax")
            c = vmap(jnp.trace)(f)
            vj = oe.contract("g,gij->ij", c, h2, backend="jax")
            vk = oe.contract("glj,gjk->lk", f, h2, backend="jax")
            vhf = vj - 0.5 * vk
            fock = h1 + vhf
            mo_energy, mo_coeff = linalg_utils._eigh(fock)
            idx = jnp.argmax(abs(mo_coeff.real), axis=0)
            mo_coeff = jnp.where(
                mo_coeff[idx, jnp.arange(len(mo_energy))].real < 0, -mo_coeff, mo_coeff
            )
            e_idx = jnp.argsort(mo_energy)
            nmo = mo_energy.size
            mo_occ = jnp.zeros(nmo)
            nocc = nelec
            mo_occ = mo_occ.at[e_idx[:nocc]].set(2)
            mocc = mo_coeff[:, jnp.nonzero(mo_occ, size=nocc)[0]]
            dm = (mocc * mo_occ[jnp.nonzero(mo_occ, size=nocc)[0]]).dot(mocc.T)
            return dm, mo_coeff

        dm0 = 2 * wave_data["mo_coeff"] @ wave_data["mo_coeff"].T.conj()
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)
        wave_data["mo_coeff"] = mo_coeff[-1][:, :nelec]
        return wave_data

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = oe.contract(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, self.norb, self.norb), 
            backend="jax")
        return ham_data
        
    @partial(jit, static_argnums=0)
    def _calc_orbenergy(self, h0, rot_h1, rot_chol, walker, wave_data=None,orbE=0):
        ene0 =0
        m = wave_data["prjlo"]
        nocc = rot_h1.shape[0]
        green_walker = self._calc_green(walker, wave_data) # in ao
        f = oe.contract('gij,jk->gik', rot_chol[:,:nocc,nocc:], green_walker.T[nocc:,:nocc], 
                        backend="jax")
        c = vmap(jnp.trace)(f)

        eneo2Jt = oe.contract('Gxk,xk,G->',f,m,c, backend="jax")*2 
        eneo2ext = oe.contract('Gxy,Gyk,xk->',f,f,m, backend="jax") 
        return eneo2Jt - eneo2ext 
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf(wave_function):
    """Class for the unrestricted Hartree-Fock wave function.

    The corresponding wave_data contains "mo_coeff", a list of two jax.Arrays of shape (norb, nelec[sigma]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        return jnp.linalg.det(
            wave_data["mo_coeff"][0].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"][1].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_green(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> list:
        """Calculates the half green's function.

        Args:
            walker_up: The walker for spin up.
            walker_dn: The walker for spin down.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function for spin up and spin down.
        """
        green_up = (
            walker_up.dot(jnp.linalg.inv(wave_data["mo_coeff"][0].T.conj() @ walker_up))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(wave_data["mo_coeff"][1].T.conj() @ walker_dn))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        fb_up = oe.contract(
            "gij,ij->g", ham_data["rot_chol"][0], green_walker[0], backend="jax"
        )
        fb_dn = oe.contract(
            "gij,ij->g", ham_data["rot_chol"][1], green_walker[1], backend="jax"
        )
        return fb_up + fb_dn

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker[0] * rot_h1[0]) \
             + jnp.sum(green_walker[1] * rot_h1[1])
        f_up = oe.contract("gij,jk->gik", rot_chol[0], green_walker[0].T,
                           backend="jax")
        f_dn = oe.contract("gij,jk->gik", rot_chol[1], green_walker[1].T,
                           backend="jax")
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (jnp.sum(c_up * c_up)
              + jnp.sum(c_dn * c_dn)
              + 2.0 * jnp.sum(c_up * c_dn)
              - exc_up - exc_dn) / 2.0

        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        dm_up = wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T.conj()
        dm_dn = wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T.conj()
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
        h1 = ham_data["h1"]
        h1 = h1.at[0].set((h1[0] + h1[0].T) / 2.0)
        h1 = h1.at[1].set((h1[1] + h1[1].T) / 2.0)
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[1], h1.shape[1]))
        nelec = self.nelec

        def scanned_fun(carry, x):
            dm = carry
            f_up = oe.contract("gij,ik->gjk", h2, dm[0], backend="jax")
            c_up = vmap(jnp.trace)(f_up)
            vj_up = oe.contract("g,gij->ij", c_up, h2, backend="jax")
            vk_up = oe.contract("glj,gjk->lk", f_up, h2, backend="jax")
            f_dn = oe.contract("gij,ik->gjk", h2, dm[1], backend="jax")
            c_dn = vmap(jnp.trace)(f_dn)
            vj_dn = oe.contract("g,gij->ij", c_dn, h2, backend="jax")
            vk_dn = oe.contract("glj,gjk->lk", f_dn, h2, backend="jax")
            fock_up = h1[0] + vj_up + vj_dn - vk_up
            fock_dn = h1[1] + vj_up + vj_dn - vk_dn
            mo_energy_up, mo_coeff_up = linalg_utils._eigh(fock_up)
            mo_energy_dn, mo_coeff_dn = linalg_utils._eigh(fock_dn)

            nmo = mo_energy_up.size

            idx_up = jnp.argmax(abs(mo_coeff_up.real), axis=0)
            mo_coeff_up = jnp.where(
                mo_coeff_up[idx_up, jnp.arange(len(mo_energy_up))].real < 0,
                -mo_coeff_up,
                mo_coeff_up,
            )
            e_idx_up = jnp.argsort(mo_energy_up)
            mo_occ_up = jnp.zeros(nmo)
            nocc_up = nelec[0]
            mo_occ_up = mo_occ_up.at[e_idx_up[:nocc_up]].set(1)
            mocc_up = mo_coeff_up[:, jnp.nonzero(mo_occ_up, size=nocc_up)[0]]
            dm_up = (mocc_up * mo_occ_up[jnp.nonzero(mo_occ_up, size=nocc_up)[0]]).dot(
                mocc_up.T
            )

            idx_dn = jnp.argmax(abs(mo_coeff_dn.real), axis=0)
            mo_coeff_dn = jnp.where(
                mo_coeff_dn[idx_dn, jnp.arange(len(mo_energy_dn))].real < 0,
                -mo_coeff_dn,
                mo_coeff_dn,
            )
            e_idx_dn = jnp.argsort(mo_energy_dn)
            mo_occ_dn = jnp.zeros(nmo)
            nocc_dn = nelec[1]
            mo_occ_dn = mo_occ_dn.at[e_idx_dn[:nocc_dn]].set(1)
            mocc_dn = mo_coeff_dn[:, jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]
            dm_dn = (mocc_dn * mo_occ_dn[jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]).dot(
                mocc_dn.T
            )

            return jnp.array([dm_up, dm_dn]), jnp.array([mo_coeff_up, mo_coeff_dn])

        dm0 = self._calc_rdm1(wave_data)
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

        wave_data["mo_coeff"] = [
            mo_coeff[-1][0][:, : nelec[0]],
            mo_coeff[-1][1][:, : nelec[1]],
        ]
        return wave_data

    @partial(jit, static_argnums=(0,))
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = [
            wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
            wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
        ]
        ham_data["rot_chol"] = [
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][0].T.conj(),
                ham_data["chol"][0].reshape(-1, self.norb, self.norb), 
                backend="jax"),
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][1].T.conj(),
                ham_data["chol"][1].reshape(-1, self.norb, self.norb), 
                backend="jax")]
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))

    
@dataclass
class cisd_ad(rhf):

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _ci_orb_olp(self, walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|C1+C2|walker>_i 
        = (C_ia <HF|i+ a|walker>/<HF|walker> + C_iajb <HF|i+ j+ a b|walker>/<HF|walker>) * <HF|walker>
        = (C_ia G_ia + C_iajb (G_ia G_jb-G_ib G_ja)) * <HF|walker>
        prj onto orbital i
        '''
        m = wave_data["prjlo"]
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
        o1 = oe.contract("ia,ka,ik->", ci1, gf[:, nocc:],m, backend="jax")
        o2 = 2 * oe.contract("iajb,ka,jb,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m, backend="jax") \
            - oe.contract("iajb,kb,ja,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m, backend="jax")
        olp = (2*o1+o2) * o0
        return olp

    # @partial(jit, static_argnums=0)
    # def _hf_ci_walker_olp_r(self,  walker: jax.Array, wave_data: dict):
    #     '''<HF|walker>/<CISD|walker>'''
    #     ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
    #     nocc = walker.shape[1]
    #     GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
    #     o1 = oe.contract("ia,ia", ci1, GF[:, nocc:], backend="jax")
    #     o2 = 2 * oe.contract("iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax") \
    #         - oe.contract("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax")
    #     olp_r = 1/(1.0 + 2 * o1 + o2)
    #     return jnp.real(olp_r)

    @partial(jit, static_argnums=0)
    def _ci_orb_olp1(self, x: float, h1_mod: jax.Array, walker: jax.Array,
                    wave_data: dict) -> complex:
        '''
        <HF|(C1+C2)_i exp(x*h1_mod)|walker>
        '''
        walker_1x = walker + x*h1_mod.dot(walker)
        olp = self._ci_orb_olp(walker_1x, wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _ci_orb_olp2(self, x: float, chol_i: jax.Array, 
                     walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|(C1+C2)_i exp(x*h2_mod)|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        olp = self._ci_orb_olp(walker_2x, wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _hf_eorb(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''hf orbital correlation energy'''
        # <HF|H_i|walker>/<HF|walker>
        rot_h1, rot_chol = ham_data['rot_h1'], ham_data['rot_chol']
        m = wave_data["prjlo"]
        nocc = rot_h1.shape[0]
        green_walker = self._calc_green(walker, wave_data)
        f = oe.contract('gij,jk->gik', rot_chol[:,:nocc,nocc:],
                        green_walker.T[nocc:,:nocc], backend="jax")
        c = vmap(jnp.trace)(f)
        eneo2Jt = oe.contract('Gxk,xk,G->',f,m,c, backend="jax")*2 
        eneo2ext = oe.contract('Gxy,Gyk,xk->',f,f,m, backend="jax")
        hf_orb_en = eneo2Jt - eneo2ext
        return jnp.real(hf_orb_en)

    @partial(jit, static_argnums=0)
    def _d2_olp2_i(self, chol_i: jax.Array,walker: jax.Array, wave_data: dict):
        x = 0.0
        f = lambda a: self._ci_orb_olp2(a,chol_i,walker,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _ci_eorb(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''
        <HF|(C1+C2)_i (H-E0)|walker>/<HF|walker>
        '''

        norb = self.norb
        chol = ham_data["chol"].reshape(-1, norb, norb)
        h1_mod = ham_data['h1_mod']
        h0_E0 = ham_data["h0"]-ham_data["E0"]

        nocc = walker.shape[1]
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2

        x = 0.0
        # one body
        f1 = lambda a: self._ci_orb_olp1(a,h1_mod,walker,wave_data)
        olp_orb12, d_overlap = jvp(f1, [x], [1.0])

        # two body
        def scanned_fun(carry, c):
            walker, wave_data = carry
            return carry, self._d2_olp2_i(c,walker,wave_data)

        _, d2_olp2_i = lax.scan(scanned_fun, (walker, wave_data), chol)
        d_2_overlap = jnp.sum(d2_olp2_i)/2

        eorb12 = (h0_E0*olp_orb12 + d_overlap + d_2_overlap) / o0

        corb12 = olp_orb12 /o0 # <(C1+C2)_i>

        return jnp.real(eorb12), jnp.real(corb12)

    @partial(jit, static_argnums=0)
    def _calc_orb_energy(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        
        eorb0 = self._hf_eorb(walker, ham_data, wave_data)
        eorb12, oorb12 = self._ci_eorb(walker, ham_data, wave_data)

        return eorb0+eorb12, eorb0, eorb12, oorb12
    
    @partial(jit, static_argnums=(0)) 
    def calc_orb_energy(self,walkers,ham_data,wave_data):
        eorb, eorb0, eorb12, oorb12 = vmap(
            self._calc_orb_energy,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return eorb, eorb0, eorb12, oorb12
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
