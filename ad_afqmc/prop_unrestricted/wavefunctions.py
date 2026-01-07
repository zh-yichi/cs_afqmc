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

class wave_function_restricted(ABC):
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

    walker_type: str
    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    # @singledispatchmethod
    # def calc_overlap(self, walkers, wave_data: dict) -> jax.Array:
    #     """Calculate the overlap < psi_t | walker > for a batch of walkers.

    #     Args:
    #         walkers : list or jax.Array
    #             The batched walkers.
    #         wave_data : dict
    #             The trial wave function data.

    #     Returns:
    #         jax.Array: The overlap of the walkers with the trial wave function.
    #     """
    #     print('walkers type', type(walkers))
    #     if
    #     raise NotImplementedError("Walker type not supported")

    # @calc_overlap.register
    # def calc_overlap(self, walkers: list, wave_data: dict) -> jax.Array:
    #     n_walkers = walkers[0].shape[0]
    #     batch_size = n_walkers // self.n_batch

    #     def scanned_fun(carry, walker_batch):
    #         walker_batch_0, walker_batch_1 = walker_batch
    #         overlap_batch = vmap(self._calc_overlap, in_axes=(0, 0, None))(
    #             walker_batch_0, walker_batch_1, wave_data
    #         )
    #         return carry, overlap_batch

    #     _, overlaps = lax.scan(
    #         scanned_fun,
    #         None,
    #         (
    #             walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
    #             walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
    #         ),
    #     )
    #     return overlaps.reshape(n_walkers)

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

    # @partial(jit, static_argnums=0)
    # def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
    #     """Overlap for a single restricted walker."""
    #     return self._calc_overlap(
    #         walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], wave_data
    #     )

    # def _calc_overlap(
    #     self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    # ) -> jax.Array:
    #     """Overlap for a single walker."""
    #     raise NotImplementedError("Overlap not defined")

    # # @singledispatchmethod
    # def calc_force_bias(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
    #     """Calculate the force bias < psi_T | chol | walker > / < psi_T | walker > for a batch of walkers.

    #     Args:
    #         walkers : list or jax.Array
    #             The batched walkers.
    #         ham_data : dict
    #             The hamiltonian data.
    #         wave_data : dict
    #             The trial wave function data.

    #     Returns:
    #         jax.Array: The force bias.
    #     """
    #     raise NotImplementedError("Walker type not supported")

    # @calc_force_bias.register
    # def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
    #     n_walkers = walkers[0].shape[0]
    #     batch_size = n_walkers // self.n_batch

    #     def scanned_fun(carry, walker_batch):
    #         walker_batch_0, walker_batch_1 = walker_batch
    #         fb_batch = vmap(self._calc_force_bias, in_axes=(0, 0, None, None))(
    #             walker_batch_0, walker_batch_1, ham_data, wave_data
    #         )
    #         return carry, fb_batch

    #     _, fbs = lax.scan(
    #         scanned_fun,
    #         None,
    #         (
    #             walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
    #             walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
    #         ),
    #     )
    #     fbs = jnp.concatenate(fbs, axis=0)
    #     return fbs.reshape(n_walkers, -1)

    # @calc_force_bias.register
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

    # @partial(jit, static_argnums=0)
    # def _calc_force_bias_restricted(
    #     self, walker: jax.Array, ham_data: dict, wave_data: dict
    # ) -> jax.Array:
    #     """Force bias for a single restricted walker."""
    #     return self._calc_force_bias(
    #         walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
    #     )

    # def _calc_force_bias(
    #     self,
    #     walker_up: jax.Array,
    #     walker_dn: jax.Array,
    #     ham_data: dict,
    #     wave_data: dict,
    # ) -> jax.Array:
    #     """Force bias for a single walker."""
    #     raise NotImplementedError("Force bias not definedr")

    # @singledispatchmethod
    # def calc_energy(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
    #     """Calculate the energy < psi_T | H | walker > / < psi_T | walker > for a batch of walkers.

    #     Args:
    #         walkers : list or jax.Array
    #             The batched walkers.
    #         ham_data : dict
    #             The hamiltonian data.
    #         wave_data : dict
    #             The trial wave function data.

    #     Returns:
    #         jax.Array: The energy.
    #     """
    #     raise NotImplementedError("Walker type not supported")

    # @calc_energy.register
    # def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
    #     n_walkers = walkers[0].shape[0]
    #     batch_size = n_walkers // self.n_batch

    #     def scanned_fun(carry, walker_batch):
    #         walker_batch_0, walker_batch_1 = walker_batch
    #         energy_batch = vmap(self._calc_energy, in_axes=(0, 0, None, None))(
    #             walker_batch_0, walker_batch_1, ham_data, wave_data
    #         )
    #         return carry, energy_batch

    #     _, energies = lax.scan(
    #         scanned_fun,
    #         None,
    #         (
    #             walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
    #             walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
    #         ),
    #     )
    #     return energies.reshape(n_walkers)

    # @calc_energy.register
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

    # @partial(jit, static_argnums=0)
    # def _calc_energy_restricted(
    #     self, walker: jax.Array, ham_data: dict, wave_data: dict
    # ) -> jax.Array:
    #     """Energy for a single restricted walker."""
    #     return self._calc_energy(
    #         walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
    #     )

    # def _calc_energy(
    #     self,
    #     walker_up: jax.Array,
    #     walker_dn: jax.Array,
    #     ham_data: dict,
    #     wave_data: dict,
    # ) -> jax.Array:
    #     """Energy for a single walker."""
    #     raise NotImplementedError("Energy not defined")

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

    # def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
    #     """Build intermediates for measurements in ham_data. This method is called by the hamiltonian class.

    #     Args:
    #         ham_data: The hamiltonian data.
    #         wave_data: The trial wave function data.

    #     Returns:
    #         ham_data: The updated Hamiltonian data.
    #     """
    #     return ham_data

    # def optimize(self, ham_data: dict, wave_data: dict) -> dict:
    #     """Optimize the wave function parameters.

    #     Args:
    #         ham_data: The hamiltonian data.
    #         wave_data: The trial wave function data.

    #     Returns:
    #         wave_data: The updated trial wave function data.
    #     """
    #     return wave_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


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
class rhf(wave_function_restricted):
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
class uhf_cpmc(uhf, wave_function_cpmc):

    @partial(jit, static_argnums=0)
    def calc_green_diagonal(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        green_up = (
            walker_up
            @ jnp.linalg.inv(wave_data["mo_coeff"][0].T.dot(walker_up))
            @ wave_data["mo_coeff"][0].T
        ).diagonal()
        green_dn = (
            walker_dn
            @ jnp.linalg.inv(wave_data["mo_coeff"][1].T.dot(walker_dn))
            @ wave_data["mo_coeff"][1].T
        ).diagonal()
        return jnp.array([green_up, green_dn])

    @partial(jit, static_argnums=0)
    def calc_green_diagonal_vmap(self, walkers: Sequence, wave_data: dict) -> jax.Array:
        return vmap(self.calc_green_diagonal, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio(
        self, green: jax.Array, update_indices: jax.Array, update_constants: jax.Array
    ) -> float:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        ratio = (1 + update_constants[0] * green[spin_i, i, i]) * (
            1 + update_constants[1] * green[spin_j, j, j]
        ) - (spin_i == spin_j) * update_constants[0] * update_constants[1] * (
            green[spin_i, i, j] * green[spin_j, j, i]
        )
        return ratio

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio_vmap(
        self, greens: jax.Array, update_indices: jax.Array, update_constants: jax.Array
    ) -> jax.Array:
        return vmap(self.calc_overlap_ratio, in_axes=(0, None, None))(
            greens, update_indices, update_constants
        )

    @partial(jit, static_argnums=0)
    def calc_full_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        green_up = (
            walker_up
            @ jnp.linalg.inv(wave_data["mo_coeff"][0].T.dot(walker_up))
            @ wave_data["mo_coeff"][0].T
        ).T
        green_dn = (
            walker_dn
            @ jnp.linalg.inv(wave_data["mo_coeff"][1].T.dot(walker_dn))
            @ wave_data["mo_coeff"][1].T
        ).T
        return jnp.array([green_up, green_dn])

    @partial(jit, static_argnums=0)
    def calc_full_green_vmap(self, walkers: Sequence, wave_data: Any) -> jax.Array:
        return vmap(self.calc_full_green, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def update_greens_function(
        self,
        green: jax.Array,
        ratio: float,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        sg_i = green[spin_i, i].at[i].add(-1)
        sg_j = green[spin_j, j].at[j].add(-1)
        g_ii = green[spin_i, i, i]
        g_jj = green[spin_j, j, j]
        g_ij = (spin_i == spin_j) * green[spin_i, i, j]
        g_ji = (spin_i == spin_j) * green[spin_j, j, i]
        green = green.at[spin_i, :, :].add(
            (update_constants[0] / ratio)
            * jnp.outer(
                green[spin_i, :, i],
                update_constants[1] * (g_ij * sg_j - g_jj * sg_i) - sg_i,
            )
        )
        green = green.at[spin_j, :, :].add(
            (update_constants[1] / ratio)
            * jnp.outer(
                green[spin_j, :, j],
                update_constants[0] * (g_ji * sg_i - g_ii * sg_j) - sg_j,
            )
        )
        return green

    @partial(jit, static_argnums=0)
    def update_greens_function_vmap(
        self, greens, ratios, update_indices, update_constants
    ):
        return vmap(self.update_greens_function, in_axes=(0, 0, None, 0))(
            greens, ratios, update_indices, update_constants
        )

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ghf(wave_function):
    """Class for the generalized Hartree-Fock wave function.

    The corresponding wave_data should contain "mo_coeff", a jax.Array of shape (2 * norb, nelec[0] + nelec[1]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            jnp.hstack(
                [
                    wave_data["mo_coeff"][: self.norb].T @ walker_up,
                    wave_data["mo_coeff"][self.norb :].T @ walker_dn,
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        overlap_mat = jnp.hstack(
            [
                wave_data["mo_coeff"][: self.norb].T @ walker_up,
                wave_data["mo_coeff"][self.norb :].T @ walker_dn,
            ]
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            jnp.vstack(
                [walker_up @ inv[: self.nelec[0]], walker_dn @ inv[self.nelec[0] :]]
            )
        ).T
        return green

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        fb = oe.contract(
            "gij,ij->g", ham_data["rot_chol"], green_walker, backend="jax"
        )
        return fb

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
        ene1 = jnp.sum(green_walker * rot_h1)
        f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
        coul = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = (jnp.sum(coul * coul) - exc) / 2.0
        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        dm = (
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]]
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T
        )
        dm_up = dm[: self.norb, : self.norb]
        dm_dn = dm[self.norb :, self.norb :]
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ jnp.block(
            [
                [ham_data["h1"][0], jnp.zeros_like(ham_data["h1"][1])],
                [jnp.zeros_like(ham_data["h1"][0]), ham_data["h1"][1]],
            ]
        )
        ham_data["rot_chol"] = vmap(
            lambda x: jnp.hstack(
                [
                    wave_data["mo_coeff"].T.conj()[:, : self.norb] @ x,
                    wave_data["mo_coeff"].T.conj()[:, self.norb :] @ x,
                ]
            ),
            in_axes=(0),
        )(ham_data["chol"].reshape(-1, self.norb, self.norb))
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ghf_cpmc(ghf, wave_function_cpmc):

    @partial(jit, static_argnums=0)
    def calc_green_diagonal(
        self, walker_up: jnp.array, walker_dn: jnp.array, wave_data: dict
    ) -> jnp.array:
        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)
        overlap_mat = (
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T @ walker_ghf
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            walker_ghf
            @ inv
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T
        ).diagonal()
        return jnp.array([green[: self.norb], green[self.norb :]])

    @partial(jit, static_argnums=0)
    def calc_green_diagonal_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        return vmap(self.calc_green_diagonal, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio(
        self, green: jnp.array, update_indices: jnp.array, update_constants: jnp.array
    ) -> float:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        i = i + (spin_i == 1) * self.norb
        j = j + (spin_j == 1) * self.norb
        ratio = (1 + update_constants[0] * green[i, i]) * (
            1 + update_constants[1] * green[j, j]
        ) - update_constants[0] * update_constants[1] * (green[i, j] * green[j, i])
        return ratio

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio_vmap(
        self, greens: jnp.array, update_indices: jnp.array, update_constants: jnp.array
    ) -> jnp.array:
        return vmap(self.calc_overlap_ratio, in_axes=(0, None, None))(
            greens, update_indices, update_constants
        )

    @partial(jit, static_argnums=0)
    def calc_full_green(
        self, walker_up: jnp.array, walker_dn: jnp.array, wave_data: dict
    ) -> jnp.array:
        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)
        green = (
            walker_ghf
            @ jnp.linalg.inv(
                wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T @ walker_ghf
            )
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T
        ).T
        return green

    @partial(jit, static_argnums=0)
    def calc_full_green_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        return vmap(self.calc_full_green, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def update_greens_function(
        self,
        green: jnp.array,
        ratio: float,
        update_indices: jnp.array,
        update_constants: jnp.array,
    ) -> jnp.array:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        i = i + (spin_i == 1) * self.norb
        j = j + (spin_j == 1) * self.norb
        sg_i = green[i].at[i].add(-1)
        sg_j = green[j].at[j].add(-1)
        green += (update_constants[0] / ratio) * jnp.outer(
            green[:, i],
            update_constants[1] * (green[i, j] * sg_j - green[j, j] * sg_i) - sg_i,
        ) + (update_constants[1] / ratio) * jnp.outer(
            green[:, j],
            update_constants[0] * (green[j, i] * sg_i - green[i, i] * sg_j) - sg_j,
        )
        return green

    @partial(jit, static_argnums=0)
    def update_greens_function_vmap(
        self, greens, ratios, update_indices, update_constants
    ):
        return vmap(self.update_greens_function, in_axes=(0, 0, None, 0))(
            greens, ratios, update_indices, update_constants
        )

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class noci(wave_function):
    """Class for the NOCI wave function.

    The corresponding wave_data should contain "ci_coeffs_dets", a list [ci_coeffs, dets]
    where ci_coeffs is a jax.Array of shape (ndets) and dets is a list [dets_up, dets_dn]
    with each being a jax.Array of shape (ndets, norb, nelec[sigma]), both ci_coeffs and dets
    are assumed to be real. The measurement methods make use of half-rotated integrals
    which are stored in ham_data (rot_h1 and rot_chol for each det).

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        ndets: Number of determinants in the NOCI expansion.
    """

    norb: int
    nelec: Tuple[int, int]
    ndets: int
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_single_det(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> jax.Array:
        """Calculate the overlap with a single determinant in the NOCI trial."""
        return jnp.linalg.det(
            trial_up[:, : self.nelec[0]].T.conj() @ walker_up
        ) * jnp.linalg.det(trial_dn[:, : self.nelec[1]].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        return jnp.sum(ci_coeffs * overlaps)

    @partial(jit, static_argnums=0)
    def _calc_green_single_det(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> List:
        """Calculate the half greens function with a single determinant in the NOCI trial."""
        green_up = (
            walker_up.dot(jnp.linalg.inv(trial_up[:, : self.nelec[0]].T.dot(walker_up)))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(trial_dn[:, : self.nelec[1]].T.dot(walker_dn)))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> Tuple:
        """Calculate the half greens function for the full trial."""
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        up_greens, dn_greens = vmap(
            self._calc_green_single_det, in_axes=(None, None, 0, 0)
        )(walker_up, walker_dn, dets[0], dets[1])
        return up_greens, dn_greens, overlaps

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        up_greens, dn_greens, overlaps = self._calc_green(
            walker_up, walker_dn, wave_data
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        fb_up = (
            oe.contract(
                "ngij,nij,n->g",
                ham_data["rot_chol"][0],
                up_greens,
                ci_coeffs * overlaps,
                backend="jax"
            )
            / overlap
        )
        fb_dn = (
            oe.contract(
                "ngij,nij,n->g",
                ham_data["rot_chol"][1],
                dn_greens,
                ci_coeffs * overlaps,
                backend="jax"
            )
            / overlap
        )
        return fb_up + fb_dn

    @partial(jit, static_argnums=0)
    def _calc_energy_single_det(
        self,
        h0: float,
        rot_h1_up: jax.Array,
        rot_h1_dn: jax.Array,
        rot_chol_up: jax.Array,
        rot_chol_dn: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> jax.Array:
        """Calculate the energy with a single determinant in the NOCI trial."""
        ene0 = h0
        green_walker = self._calc_green_single_det(
            walker_up, walker_dn, trial_up, trial_dn
        )
        ene1 = jnp.sum(green_walker[0] * rot_h1_up) + jnp.sum(
            green_walker[1] * rot_h1_dn
        )
        f_up = oe.contract(
            "gij,jk->gik", rot_chol_up, green_walker[0].T, backend="jax")
        f_dn = oe.contract(
            "gij,jk->gik", rot_chol_dn, green_walker[1].T, backend="jax")
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (
            jnp.sum(c_up * c_up)
            + jnp.sum(c_dn * c_dn)
            + 2.0 * jnp.sum(c_up * c_dn)
            - exc_up
            - exc_dn
        ) / 2.0

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
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        energies = vmap(
            self._calc_energy_single_det, in_axes=(None, 0, 0, 0, 0, None, None, 0, 0)
        )(
            h0,
            rot_h1[0],
            rot_h1[1],
            rot_chol[0],
            rot_chol[1],
            walker_up,
            walker_dn,
            dets[0],
            dets[1],
        )
        ene = jnp.sum(ci_coeffs * overlaps * energies) / overlap
        return ene

    @partial(jit, static_argnums=0)
    def _get_trans_rdm1_single_det(self, sd_0_up, sd_0_dn, sd_1_up, sd_1_dn) -> list:
        dm_up = (
            (sd_0_up[:, : self.nelec[0]])
            .dot(
                jnp.linalg.inv(
                    sd_1_up[:, : self.nelec[0]].T.dot(sd_0_up[:, : self.nelec[0]])
                )
            )
            .dot(sd_1_up[:, : self.nelec[0]].T)
        )
        dm_dn = (
            (sd_0_dn[:, : self.nelec[1]])
            .dot(
                jnp.linalg.inv(
                    sd_1_dn[:, : self.nelec[1]].T.dot(sd_0_dn[:, : self.nelec[1]])
                )
            )
            .dot(sd_1_dn[:, : self.nelec[1]].T)
        )
        return [dm_up, dm_dn]

    @partial(jit, static_argnums=0)
    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(
            vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0)),
            in_axes=(0, 0, None, None),
        )(dets[0], dets[1], dets[0], dets[1])
        overlap = jnp.sum(jnp.outer(ci_coeffs, ci_coeffs) * overlaps)
        up_rdm1s, dn_rdm1s = vmap(
            vmap(self._get_trans_rdm1_single_det, in_axes=(0, 0, None, None)),
            in_axes=(None, None, 0, 0),
        )(dets[0], dets[1], dets[0], dets[1])
        up_rdm1 = (
            oe.contract(
                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, up_rdm1s, 
                backend="jax") / overlap
        )
        dn_rdm1 = (
            oe.contract(
                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, dn_rdm1s, 
                backend="jax") / overlap
        )
        return jnp.array([up_rdm1, dn_rdm1])

    @partial(jit, static_argnums=(0,))
    def _rot_orbs_single_det(
        self, ham_data: dict, trial_up: jax.Array, trial_dn: jax.Array
    ) -> Tuple:
        rot_h1 = [
            trial_up.T.conj() @ ham_data["h1"][0],
            trial_dn.T.conj() @ ham_data["h1"][1],
        ]
        rot_chol = [
            oe.contract(
                "pi,gij->gpj",
                trial_up.T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
                backend="jax"),
            oe.contract(
                "pi,gij->gpj",
                trial_dn.T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
                backend="jax")]
        return rot_h1, rot_chol

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        trial = wave_data["ci_coeffs_dets"][1]
        ham_data["rot_h1"], ham_data["rot_chol"] = vmap(
            self._rot_orbs_single_det, in_axes=(None, 0, 0)
        )(ham_data, trial[0], trial[1])
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class wave_function_auto(wave_function):
    """This wave function only requires the definition of overlap functions.
    It evaluates force bias and local energy by differentiating various overlaps
    (single derivatives with AD and double with FD)."""

    def __post_init__(self):
        """eps is the finite difference step size in local energy calculations."""
        if not hasattr(self, "eps"):
            self.eps = 1.0e-4

    @partial(jit, static_argnums=0)
    def _overlap_with_rot_sd_restricted(
        self,
        x_gamma: jax.Array,
        walker: jax.Array,
        chol: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        """Helper function for calculating force bias using AD,
        evaluates < psi_T | exp(x_gamma * chol) | walker > to linear order"""
        x_chol = oe.contract(
            "gij,g->ij", chol.reshape(-1, self.norb, self.norb), x_gamma, backend="jax"
        )
        walker_1 = walker + x_chol.dot(walker)
        return self._calc_overlap_restricted(walker_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker > by differentiating
        < psi_T | exp(x_gamma * chol) | walker > / < psi_T | walker >"""
        val, grad = vjp(
            self._overlap_with_rot_sd_restricted,
            jnp.zeros((ham_data["chol"].shape[0],)) + 0.0j,
            walker,
            ham_data["chol"],
            wave_data,
        )
        return grad(1.0 + 0.0j)[0] / val

    @partial(jit, static_argnums=0)
    def _overlap_with_rot_sd(
        self,
        x_gamma: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        chol: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating force bias using AD,
        evaluates < psi_T | exp(x_gamma * chol) | walker > to linear order"""
        x_chol = oe.contract(
            "gij,g->ij", chol.reshape(-1, self.norb, self.norb), x_gamma, backend="jax"
        )
        walker_up_1 = walker_up + x_chol.dot(walker_up)
        walker_dn_1 = walker_dn + x_chol.dot(walker_dn)
        return self._calc_overlap(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: Any,
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker > by differentiating
        < psi_T | exp(x_gamma * chol) | walker > / < psi_T | walker >"""
        val, grad = vjp(
            self._overlap_with_rot_sd,
            jnp.zeros((ham_data["chol"].shape[0],)) + 0.0j,
            walker_up,
            walker_dn,
            ham_data["chol"],
            wave_data,
        )
        return grad(1.0 + 0.0j)[0] / val

    @partial(jit, static_argnums=0)
    def _overlap_with_single_rot_restricted(
        self, x: float, h1: jax.Array, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walker_2 = walker + x * h1.dot(walker)
        return self._calc_overlap_restricted(walker_2, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot_restricted(
        self, x: float, chol_i: jax.Array, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * chol_i) | walker > to quadratic order"""
        walker2 = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
        return self._calc_overlap_restricted(walker2, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self,
        walker: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        """Calculates local energy using AD and finite difference for the two body term"""

        h0, h1, chol, v0 = (
            ham_data["h0"],
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0,
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["normal_ordering_term"],
        )

        x = 0.0
        # one body
        f1 = lambda a: self._overlap_with_single_rot_restricted(
            a, h1 + v0, walker, wave_data
        )
        overlap, d_overlap = jvp(f1, [x], [1.0])

        # two body
        eps = self.eps

        # carry: [eps, walker, wave_data]
        def scanned_fun(carry, x):
            eps, walker, wave_data = carry
            return carry, self._overlap_with_double_rot_restricted(
                eps, x, walker, wave_data
            )

        _, overlap_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, overlap_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, overlap_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

        return (d_overlap + jnp.sum(d_2_overlap) / 2.0) / overlap + h0

    @partial(jit, static_argnums=0)
    def _overlap_with_single_rot(
        self,
        x: float,
        h1: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walker_up_1 = walker_up + x * h1[0].dot(walker_up)
        walker_dn_1 = walker_dn + x * h1[1].dot(walker_dn)
        return self._calc_overlap(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot(
        self,
        x: float,
        chol_i: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * chol_i) | walker > to quadratic order"""
        walker_up_1 = (
            walker_up
            + x * chol_i.dot(walker_up)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker_up))
        )
        walker_dn_1 = (
            walker_dn
            + x * chol_i.dot(walker_dn)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker_dn))
        )
        return self._calc_overlap(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: Any,
    ) -> jax.Array:
        """Calculates local energy using AD and finite difference for the two body term"""

        h0, h1, chol, v0 = (
            ham_data["h0"],
            ham_data["h1"],
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["normal_ordering_term"],
        )

        x = 0.0
        # one body
        f1 = lambda a: self._overlap_with_single_rot(
            a, h1 + v0, walker_up, walker_dn, wave_data
        )
        val1, dx1 = jvp(f1, [x], [1.0])

        # two body
        # vmap_fun = vmap(
        #     self._overlap_with_double_rot, in_axes=(None, 0, None, None, None)
        # )

        eps = self.eps

        # carry: [eps, walker, wave_data]
        def scanned_fun(carry, chol_i):
            eps, walker_up, walker_dn, wave_data = carry
            return carry, self._overlap_with_double_rot(
                eps, chol_i, walker_up, walker_dn, wave_data
            )

        _, overlap_p = lax.scan(
            scanned_fun, (eps, walker_up, walker_dn, wave_data), chol
        )
        _, overlap_0 = lax.scan(
            scanned_fun, (0.0, walker_up, walker_dn, wave_data), chol
        )
        _, overlap_m = lax.scan(
            scanned_fun, (-1.0 * eps, walker_up, walker_dn, wave_data), chol
        )
        d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

        # dx2 = (
        #     (
        #         vmap_fun(eps, chol, walker_up, walker_dn, wave_data)
        #         - 2.0 * vmap_fun(zero, chol, walker_up, walker_dn, wave_data)
        #         + vmap_fun(-1.0 * eps, chol, walker_up, walker_dn, wave_data)
        #     )
        #     / eps
        #     / eps
        # )

        return (dx1 + jnp.sum(d_2_overlap) / 2.0) / val1 + h0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        v0 = 0.5 * oe.contract(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            backend="jax")
        ham_data["normal_ordering_term"] = -v0
        return ham_data


@dataclass
class multislater(wave_function_auto):
    """Multislater wave function implemented using the auto class.

    We work in the orbital basis of the wave function.
    Associated wave_data consists of excitation indices and ci coefficients:
        Acre: Alpha creation indices.
        Ades: Alpha destruction indices.
        Bcre: Beta creation indices.
        Bdes: Beta destruction indices.
        coeff: Coefficients of the determinants.
        ref_det: Reference determinant.
    """

    norb: int
    nelec: Tuple[int, int]
    max_excitation: int  # maximum of sum of alpha and beta excitation ranks
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _det_overlap(
        self, green: jax.Array, cre: jax.Array, des: jax.Array
    ) -> jax.Array:
        return jnp.linalg.det(green[jnp.ix_(cre, des)])

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        ref_det = wave_data["ref_det"][0]
        return (
            walker.dot(
                jnp.linalg.inv(walker[jnp.nonzero(ref_det, size=self.nelec[0])[0], :])
            )
        ).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        """Calclulates < psi_T | walker > efficiently using Wick's theorem"""
        Acre, Ades, Bcre, Bdes, coeff, ref_det = (
            wave_data["Acre"],
            wave_data["Ades"],
            wave_data["Bcre"],
            wave_data["Bdes"],
            wave_data["coeff"],
            wave_data["ref_det"],
        )
        green = self._calc_green_restricted(walker, wave_data)

        # overlap with the reference determinant
        overlap_0 = (
            jnp.linalg.det(walker[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :])
            ** 2
        )

        # overlap / overlap_0
        overlap = coeff[(0, 0)] + 0.0j

        for i in range(1, self.max_excitation + 1):
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green, Acre[(i, 0)], Ades[(i, 0)]
            ).dot(coeff[(i, 0)])
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green, Bcre[(0, i)], Bdes[(0, i)]
            ).dot(coeff[(0, i)])

            for j in range(1, self.max_excitation - i + 1):
                overlap_a = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green, Acre[(i, j)], Ades[(i, j)]
                )
                overlap_b = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green, Bcre[(i, j)], Bdes[(i, j)]
                )
                overlap += (overlap_a * overlap_b) @ coeff[(i, j)]

        return (overlap * overlap_0)[0]

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> list:
        ref_det = wave_data["ref_det"]
        green_up = (
            walker_up.dot(
                jnp.linalg.inv(
                    walker_up[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :]
                )
            )
        ).T
        green_dn = (
            walker_dn.dot(
                jnp.linalg.inv(
                    walker_dn[jnp.nonzero(ref_det[1], size=self.nelec[1])[0], :]
                )
            )
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        """Calclulates < psi_T | walker > efficiently using Wick's theorem"""
        Acre, Ades, Bcre, Bdes, coeff, ref_det = (
            wave_data["Acre"],
            wave_data["Ades"],
            wave_data["Bcre"],
            wave_data["Bdes"],
            wave_data["coeff"],
            wave_data["ref_det"],
        )
        green = self._calc_green(walker_up, walker_dn, wave_data)

        # overlap with the reference determinant
        overlap_0 = jnp.linalg.det(
            walker_up[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :]
        ) * jnp.linalg.det(walker_dn[jnp.nonzero(ref_det[1], size=self.nelec[1])[0], :])

        # overlap / overlap_0
        overlap = coeff[(0, 0)] + 0.0j

        for i in range(1, self.max_excitation + 1):
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green[0], Acre[(i, 0)], Ades[(i, 0)]
            ).dot(coeff[(i, 0)])
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green[1], Bcre[(0, i)], Bdes[(0, i)]
            ).dot(coeff[(0, i)])

            for j in range(1, self.max_excitation - i + 1):
                overlap_a = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green[0], Acre[(i, j)], Ades[(i, j)]
                )
                overlap_b = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green[1], Bcre[(i, j)], Bdes[(i, j)]
                )
                overlap += (overlap_a * overlap_b) @ coeff[(i, j)]

        return (overlap * overlap_0)[0]

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        """Spatial 1RDM of the reference det"""
        ref_det = wave_data["ref_det"]
        orb_mat = np.eye(self.norb)
        orbs_up = orb_mat[:, ref_det[0] > 0]
        orbs_dn = orb_mat[:, ref_det[1] > 0]
        rdm_up = orbs_up @ orbs_up.T
        rdm_dn = orbs_dn @ orbs_dn.T
        return jnp.array([rdm_up, rdm_dn])

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class CISD(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = self._calc_green_restricted(walker)
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = oe.contract("ia,ia", ci1, GF[:, nocc:], backend="jax")
        o2 = 2 * oe.contract(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return (1.0 + 2 * o1 + o2) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class UCISD(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array
    ) -> List[jax.Array]:

        green_up = (walker_up.dot(jnp.linalg.inv(walker_up[: walker_up.shape[1], :]))).T
        green_dn = (walker_dn.dot(jnp.linalg.inv(walker_dn[: walker_dn.shape[1], :]))).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:

        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]
        # moA, moB = wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]
        # walker_dn_B = moB.T.dot(
        #     walker_dn[:, :noccB]
        # )  # put walker_dn in the basis of alpha reference

        GFA, GFB = self._calc_green(walker_up, walker_dn)

        o0 = jnp.linalg.det(walker_up[:noccA, :]) * jnp.linalg.det(walker_dn[:noccB, :])

        o1 = oe.contract("ia,ia", ci1A, GFA[:, noccA:], backend="jax") + oe.contract(
            "ia,ia", ci1B, GFB[:, noccB:], backend="jax"
        )

        # AA
        o2 = 0.5 * oe.contract("iajb, ia, jb", ci2AA, GFA[:, noccA:], GFA[:, noccA:], backend="jax")
        # BB
        o2 += 0.5 * oe.contract("iajb, ia, jb", ci2BB, GFB[:, noccB:], GFB[:, noccB:], backend="jax")
        # AB
        o2 += oe.contract("iajb, ia, jb", ci2AB, GFA[:, noccA:], GFB[:, noccB:], backend="jax")

        return (1.0 + o1 + o2) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class GCISD(wave_function_auto):
    """This class contains functions for the GCISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>
    The wave_data need to store the coefficients C(ia) and C(ia jb) and the GHF mo coefficients
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        nocc, ci1, ci2 = (
            self.nelec[0] + self.nelec[1],
            wave_data["ci1"],
            wave_data["ci2"],
        )
        walker = jnp.block(
            [
                [walker_up, jnp.zeros_like(walker_dn)],
                [jnp.zeros_like(walker_up), walker_dn],
            ]
        )
        walker = wave_data["mo_coeff"].T @ walker
        GF = self._calc_green(walker)
        o0 = jnp.linalg.det(walker[: walker.shape[1], :])
        o1 = oe.contract("ia,ia", ci1, GF[:, nocc:], backend="jax")
        o2 = oe.contract("iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax") \
            - oe.contract("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return (1.0 + o1 + o2 / 4.0) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class CISD_THC(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)  in the THC format
    i.e. C(i,a,j,b) = X1(P,i) X2(P,a) V(P,Q) X1(P,j) X2(P,b)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, Xocc, Xvirt, VKL = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["Xocc"],
            wave_data["Xvirt"],
            wave_data["VKL"],
        )
        GF = self._calc_green_restricted(walker)

        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2

        o1 = oe.contract("ia,ia", ci1, GF[:, nocc:], backend="jax")

        # A = jnp.einsum('ia,Pi,Pa->P', GF[:,nocc:], Xocc, Xvirt)
        # o2 = 2*jnp.einsum('P,PQ,Q', A, VKL, A)

        gv = GF[:, nocc:] @ Xvirt.T
        A = oe.contract("Pi,iP->P", Xocc, gv, backend="jax")
        o2 = 2 * (A @ VKL).dot(A)

        B = Xocc @ gv
        o2 -= jnp.sum(B * B.T * VKL)

        return (1.0 + 2 * o1 + o2) * o0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_hf1(rhf):
    """
    use CISD Trial and HF Guide. The 1st way: split the 
    overlap ratio of Trial and Guide into an extra observable

    w(walker) = weight accumulated by HF importance sampling
    E_local(walker) = <CISD|H|walker>/<HF|walker>
    O_local(walker) = <CISD|wallker>/<HF|walker>
    <CISD|H|walkers>/<CISD|walkers>
      = <E>/<O>
      = {sum_walker w(walker)*E_local(walker) / sum w(walker)} 
        * {sum_walker w(walker)*O_local(walker) / sum w(walker)}^(-1) 
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    
    @partial(jit, static_argnums=0)
    def _calc_energy_cisd_hf(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        # rot_chol = ham_data["rot_chol"]
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        h0 = ham_data["h0"]

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
        # e1 = e1_0 + e1_1 + e1_2

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
        lci1g = oe.contract("gip,qi->gpq", ham_data["lci1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
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

        # e2 = e2_0 + e2_1 + e2_2
        e0 = e1_0 + e2_0 # <HF|(h1+h2)|phi>/<HF|phi>
        e12 = e1_1 + e1_2 + e2_1 + e2_2 # <HF|(c1+c2)(h1+h2)|phi>/<HF|phi>
        olp = 1 + 2*ci1g + gci2g

        return jnp.real(olp), jnp.real(h0+e0), jnp.real(e0+e12)

    @partial(jit, static_argnums=(0))
    def calc_energy_cisd_hf(self,walkers,ham_data,wave_data):
        olp, e0, e12 = vmap(
            self._calc_energy_cisd_hf,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return olp, e0, e12

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        norb = self.norb
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

@dataclass
class cisd_hf2(cisd_hf1):
    """
    use CISD Trial and HF Guide, the 2nd way: 
    abosrb the overlap ratio <Trial|walker>/<Guide/walker> into the weight

    w'(walker)  = weight (for measurements) 
                = weight accumulated by HF importance sampling * <CISD|walker>/<HF|walker>
    E_local(walker) = <CISD|H|walker>/<CISD|walker>
    <E> = {sum_walker w'(walker) * E_local(walker)} / {sum_walker w'(walker)}
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    
    @partial(jit, static_argnums=0)
    def _calc_energy_cisd_hf(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        # rot_chol = ham_data["rot_chol"]
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        h0 = ham_data["h0"]

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
        # e1 = e1_0 + e1_1 + e1_2

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
        lci1g = oe.contract("gip,qi->gpq", ham_data["lci1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
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

        olp = 1 + 2*ci1g + gci2g # <CISD|walker>/HF|walker>
        e_hf = h0 + e1_0 + e2_0 # <HF|(h1+h2)|phi>/<HF|phi>
        # h0 + {<HF|(1+C1+C2)(h1+h2)|walker>/<HF|walker>} / {<CISD|walker>/<HF|walker>}
        e_ci = h0 + (e1_0 + e2_0 + e1_1 + e1_2 + e2_1 + e2_2) / olp

        return jnp.real(olp), jnp.real(e_hf), jnp.real(e_ci)

    @partial(jit, static_argnums=(0))
    def calc_energy_cisd_hf(self,walkers,ham_data,wave_data):
        olp, ehf, eci = vmap(
            self._calc_energy_cisd_hf,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return olp, ehf, eci


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

@dataclass
class cisd_pt(wave_function):
    """A manual implementation of the CISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = oe.contract("ia,ia", ci1, GF[:, nocc:], backend="jax")
        o2 = 2 * oe.contract(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return (1.0 + 2 * o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")

        # ref
        fb_0 = 2 * lg

        # single excitations
        ci1g = oe.contract("pt,pt->", ci1, green_occ, backend="jax")
        ci1gp = oe.contract("pt,it->pi", ci1, greenp, backend="jax")
        gci1gp = oe.contract("pj,pi->ij", green, ci1gp, backend="jax")
        fb_1_1 = 4 * ci1g * lg
        fb_1_2 = -2 * oe.contract("gij,ij->g", chol, gci1gp, backend="jax")
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_c = oe.contract("ptqu,pt->qu", ci2, green_occ, backend="jax")
        ci2g_e = oe.contract("ptqu,pu->qt", ci2, green_occ, backend="jax")
        cisd_green_c = (greenp @ ci2g_c.T) @ green
        cisd_green_e = (greenp @ ci2g_e.T) @ green
        cisd_green = -4 * cisd_green_c + 2 * cisd_green_e
        ci2g = 4 * ci2g_c - 2 * ci2g_e
        gci2g = oe.contract("qu,qu->", ci2g, green_occ, backend="jax")
        fb_2_1 = lg * gci2g
        fb_2_2 = oe.contract("gij,ij->g", chol, cisd_green, backend="jax")
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g / 2.0
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        # rot_chol = ham_data["rot_chol"]
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        h0 = ham_data["h0"]

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
        lci1g = oe.contract("gip,qi->gpq", ham_data["lci1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
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

        e1 = e1_1 + e1_2
        e2 = e2_1 + e2_2

        e0 = e1_0 + e2_0
        e12 = e1 + e2

        # overlap
        o12 = 2 * ci1g + gci2g

        # E0 = h0 + e0
        # E1 = e12 - o12*e0
        # E2 = -o12 * (e12 - o12*e0)
        # E3 = o12**2 * (e12 - o12*e0)
        # E4 = -o12**3 * (e12 - o12*e0)
        # E5 = o12**4 * (e12 - o12*e0)
        # E6 = -o12**5 * (e12 - o12*e0)
        # E7 = o12**6 * (e12 - o12*e0)
        # E8 = -o12**7 * (e12 - o12*e0)

        e_cisd = h0 + e0 + (e12-o12*e0)/(1+o12)

        return e_cisd #E0+E1+E2+E3+E4+E5+E6+E7+E8


    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["lci1"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            backend="jax"
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd(wave_function):
    """A manual implementation of the CISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = oe.contract("ia,ia", ci1, GF[:, nocc:], backend="jax")
        o2 = 2 * oe.contract(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return (1.0 + 2 * o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")

        # ref
        fb_0 = 2 * lg

        # single excitations
        ci1g = oe.contract("pt,pt->", ci1, green_occ, backend="jax")
        ci1gp = oe.contract("pt,it->pi", ci1, greenp, backend="jax")
        gci1gp = oe.contract("pj,pi->ij", green, ci1gp, backend="jax")
        fb_1_1 = 4 * ci1g * lg
        fb_1_2 = -2 * oe.contract("gij,ij->g", chol, gci1gp, backend="jax")
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_c = oe.contract("ptqu,pt->qu", ci2, green_occ, backend="jax")
        ci2g_e = oe.contract("ptqu,pu->qt", ci2, green_occ, backend="jax")
        cisd_green_c = (greenp @ ci2g_c.T) @ green
        cisd_green_e = (greenp @ ci2g_e.T) @ green
        cisd_green = -4 * cisd_green_c + 2 * cisd_green_e
        ci2g = 4 * ci2g_c - 2 * ci2g_e
        gci2g = oe.contract("qu,qu->", ci2g, green_occ, backend="jax")
        fb_2_1 = lg * gci2g
        fb_2_2 = oe.contract("gij,ij->g", chol, cisd_green, backend="jax")
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g / 2.0
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        # rot_chol = ham_data["rot_chol"]
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        e0 = ham_data["h0"]

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
        lci1g = oe.contract("gip,qi->gpq", ham_data["lci1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
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
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0


    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["lci1"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            backend="jax"
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
   

@dataclass
class cisd_faster(cisd):
    """A manual implementation of the CISD wave function.

    Faster than cisd, but the energy function builds some large intermediates, O(XMN),
    so memory usage is high.
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        ci1g = oe.contract("pt,pt->", ci1, green_occ, backend="jax")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 *oe.contract("ij,ij->", h1, ci1_green, backend="jax")
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
        lg1 = oe.contract("gpj,qj->gpq", rot_chol, green, backend="jax")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = oe.contract("gij,ij->g", chol, ci1_green, backend="jax")
        e2_1_2 = -2 * (lci1g @ lg)
        gl = oe.contract("pj,gji->gpi", green, chol, backend="jax")
        ci1g1 = ci1 @ green_occ.T
        e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg1, lg1, ci1g1, backend="jax")
        lci1g = oe.contract("gip,qi->gpq", ham_data["lci1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg
        lci2_green = oe.contract("gpi,ji->gpj", rot_chol, ci2_green, backend="jax")
        e2_2_2_2 = 0.5 * oe.contract("gpi,gpi->", gl, lci2_green, backend="jax")
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        glgp = oe.contract("gpi,it->gpt", gl, greenp, backend="jax")
        l2ci2_1 = oe.contract("gpt,gqu,ptqu->g", glgp, glgp, ci2, backend="jax")
        l2ci2_2 = oe.contract("gpu,gqt,ptqu->g", glgp, glgp, ci2, backend="jax")
        e2_2_3 = 2 * l2ci2_1.sum() - l2ci2_2.sum()
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3
        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0
     

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ucisd(wave_function):
    """A manual implementation of the UCISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        noccA, noccB = self.nelec[0], self.nelec[1]
        dm_up = (wave_data["mo_coeff"][0][:,:noccA] 
                 @ wave_data["mo_coeff"][0][:,:noccA].T.conj())
        dm_dn = (wave_data["mo_coeff"][1][:,:noccB] 
                 @ wave_data["mo_coeff"][1][:,:noccB].T.conj())
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[: noccA, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[: noccB, :]))).T
        green_a, green_b = green_a[:, noccA:], green_b[:, noccB:]
        o0 = jnp.linalg.det(walker_up[:noccA, :]) * jnp.linalg.det(walker_dn[:noccB, :])
        o1 = oe.contract("ia,ia", ci1A, green_a, backend="jax") \
            + oe.contract("ia,ia", ci1B, green_b, backend="jax")
        o2 = 0.5 * oe.contract("iajb, ia, jb", ci2AA, green_a, green_a, backend="jax")\
            + 0.5 * oe.contract("iajb, ia, jb", ci2BB, green_b, green_b, backend="jax")\
            + oe.contract("iajb, ia, jb", ci2AB, green_a, green_b, backend="jax")
        return (1.0 + o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        nocc_a, ci1_a, ci2_aa = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        nocc_b, ci1_b, ci2_bb = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2_ab = wave_data["ci2AB"]
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, : self.nelec[0], :]
        rot_chol_b = chol_b[:, : self.nelec[1], :]
        lg_a = oe.contract("gpj,pj->g", rot_chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpj,pj->g", rot_chol_b, green_b, backend="jax")
        lg = lg_a + lg_b

        # ref
        fb_0 = lg_a + lg_b

        # single excitations
        ci1g_a = oe.contract("pt,pt->", ci1_a, green_occ_a, backend="jax")
        ci1g_b = oe.contract("pt,pt->", ci1_b, green_occ_b, backend="jax")
        ci1g = ci1g_a + ci1g_b
        fb_1_1 = ci1g * lg
        ci1gp_a = oe.contract("pt,it->pi", ci1_a, greenp_a, backend="jax")
        ci1gp_b = oe.contract("pt,it->pi", ci1_b, greenp_b, backend="jax")
        gci1gp_a = oe.contract("pj,pi->ij", green_a, ci1gp_a, backend="jax")
        gci1gp_b = oe.contract("pj,pi->ij", green_b, ci1gp_b, backend="jax")
        fb_1_2 = -oe.contract(
            "gij,ij->g", chol_a, gci1gp_a, backend="jax")\
                - oe.contract("gij,ij->g", chol_b, gci1gp_b, backend="jax")
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_a = oe.contract("ptqu,pt->qu", ci2_aa, green_occ_a, backend="jax")
        ci2g_b = oe.contract("ptqu,pt->qu", ci2_bb, green_occ_b, backend="jax")
        ci2g_ab_a = oe.contract("ptqu,qu->pt", ci2_ab, green_occ_b, backend="jax")
        ci2g_ab_b = oe.contract("ptqu,pt->qu", ci2_ab, green_occ_a, backend="jax")
        gci2g_a = 0.5 * oe.contract("qu,qu->", ci2g_a, green_occ_a, backend="jax")
        gci2g_b = 0.5 * oe.contract("qu,qu->", ci2g_b, green_occ_b, backend="jax")
        gci2g_ab = oe.contract("pt,pt->", ci2g_ab_a, green_occ_a, backend="jax")
        gci2g = gci2g_a + gci2g_b + gci2g_ab
        fb_2_1 = lg * gci2g
        ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
        ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
        fb_2_2_a = -oe.contract("gij,ij->g", chol_a, ci2_green_a, backend="jax")
        fb_2_2_b = -oe.contract("gij,ij->g", chol_b, ci2_green_b, backend="jax")
        fb_2_2 = fb_2_2_a + fb_2_2_b
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a, ci1_a, ci2_aa = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        nocc_b, ci1_b, ci2_bb = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2_ab = wave_data["ci2AB"]
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]
        h1_a = ham_data["h1"][0]
        h1_b = ham_data["h1"][1]
        hg_a = oe.contract("pj,pj->", h1_a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1_b[:nocc_b, :], green_b, backend="jax")
        hg = hg_a + hg_b

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = hg

        # single excitations
        ci1g_a = oe.contract("pt,pt->", ci1_a, green_occ_a, backend="jax")
        ci1g_b = oe.contract("pt,pt->", ci1_b, green_occ_b, backend="jax")
        ci1g = ci1g_a + ci1g_b
        e1_1_1 = ci1g * hg
        gpci1_a = greenp_a @ ci1_a.T
        gpci1_b = greenp_b @ ci1_b.T
        ci1_green_a = gpci1_a @ green_a
        ci1_green_b = gpci1_b @ green_b
        e1_1_2 = -(
            oe.contract("ij,ij->", h1_a, ci1_green_a, backend="jax")
            + oe.contract("ij,ij->", h1_b, ci1_green_b, backend="jax")
        )
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_a = oe.contract("ptqu,pt->qu", ci2_aa, green_occ_a, backend="jax") / 4
        ci2g_b = oe.contract("ptqu,pt->qu", ci2_bb, green_occ_b, backend="jax") / 4
        ci2g_ab_a = oe.contract("ptqu,qu->pt", ci2_ab, green_occ_b, backend="jax")
        ci2g_ab_b = oe.contract("ptqu,pt->qu", ci2_ab, green_occ_a, backend="jax")
        gci2g_a = oe.contract("qu,qu->", ci2g_a, green_occ_a, backend="jax")
        gci2g_b = oe.contract("qu,qu->", ci2g_b, green_occ_b, backend="jax")
        gci2g_ab = oe.contract("pt,pt->", ci2g_ab_a, green_occ_a, backend="jax")
        gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
        e1_2_1 = hg * gci2g
        ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
        ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
        ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
        ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
        e1_2_2_a = -oe.contract(
            "ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a, backend="jax")
        e1_2_2_b = -oe.contract(
            "ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2

        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg_a = oe.contract("gpj,pj->g", rot_chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpj,pj->g", rot_chol_b, green_b, backend="jax")
        e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
        lg1_a = oe.contract("gpj,qj->gpq", rot_chol_a, green_a, backend="jax")
        lg1_b = oe.contract("gpj,qj->gpq", rot_chol_b, green_b, backend="jax")
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
        lci1g_a = oe.contract("gij,ij->g", chol_a, ci1_green_a, backend="jax")
        lci1g_b = oe.contract("gij,ij->g", chol_b, ci1_green_b, backend="jax")
        e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
        ci1g1_a = ci1_a @ green_occ_a.T
        ci1g1_b = ci1_b @ green_occ_b.T
        e2_1_3_1 = oe.contract(
            "gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, backend="jax"
        ) + oe.contract("gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, backend="jax")
        lci1g_a = oe.contract(
            "gip,qi->gpq", ham_data["lci1_a"], green_a, backend="jax"
        )
        lci1g_b = oe.contract(
            "gip,qi->gpq", ham_data["lci1_b"], green_b, backend="jax"
        )
        e2_1_3_2 = -oe.contract(
            "gpq,gqp->", lci1g_a, lg1_a, backend="jax"
        ) - oe.contract("gpq,gqp->", lci1g_b, lg1_b, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + e2_1_2 + e2_1_3

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g_a = oe.contract("gij,ij->g",
            chol_a, 8 * ci2_green_a + 2 * ci2_green_ab_a, backend="jax")
        lci2g_b = oe.contract("gij,ij->g",
            chol_b, 8 * ci2_green_b + 2 * ci2_green_ab_b, backend="jax")
        e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = oe.contract("pj,ji->pi", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("pj,ji->pi", green_b, chol_b_i, backend="jax")
            lci2_green_a_i = oe.contract(
                "pi,ji->pj",
                rot_chol_a_i,
                8 * ci2_green_a + 2 * ci2_green_ab_a, backend="jax"
            )
            lci2_green_b_i = oe.contract(
                "pi,ji->pj",
                rot_chol_b_i,
                8 * ci2_green_b + 2 * ci2_green_ab_b, backend="jax"
            )
            carry[0] += 0.5 * (
                oe.contract("pi,pi->", gl_a_i, lci2_green_a_i, backend="jax")
                + oe.contract("pi,pi->", gl_b_i, lci2_green_b_i, backend="jax")
            )
            glgp_a_i = oe.contract(
                "pi,it->pt", gl_a_i, greenp_a, backend="jax"
            )
            glgp_b_i = oe.contract(
                "pi,it->pt", gl_b_i, greenp_b, backend="jax"
            )
            l2ci2_a = 0.5 * oe.contract(
                "pt,qu,ptqu->",
                glgp_a_i, glgp_a_i, ci2_aa, backend="jax")
            l2ci2_b = 0.5 * oe.contract(
                "pt,qu,ptqu->",
                glgp_b_i, glgp_b_i, ci2_bb, backend="jax")
            l2ci2_ab = oe.contract(
                "pt,qu,ptqu->",
                glgp_a_i, glgp_b_i, ci2_ab, backend="jax")
            carry[1] += l2ci2_a + l2ci2_b + l2ci2_ab
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["lci1_a"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"][0].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1A"],
            backend="jax")
        ham_data["lci1_b"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"][1].reshape(-1, self.norb, self.norb)[:, :, self.nelec[1] :],
            wave_data["ci1B"],
            backend="jax")
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt(rhf):
    """A manual implementation of the CCSD_PT wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        nocc = self.nelec[0]
        gf = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        return gf

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        t1, t2 = wave_data["t1"], wave_data["t2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        h0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        t1g = oe.contract("pt,pt->", t1, green_occ, backend="jax")
        e1_1_1 = 4 * t1g * hg
        gpt1 = greenp @ t1.T
        t1_green = gpt1 @ green
        e1_1_2 = -2 * oe.contract("ij,ij->", h1, t1_green, backend="jax")
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        t2g_c = oe.contract("ptqu,pt->qu", t2, green_occ, backend="jax")
        t2g_e = oe.contract("ptqu,pu->qt", t2, green_occ, backend="jax")
        t2_green_c = (greenp @ t2g_c.T) @ green
        t2_green_e = (greenp @ t2g_e.T) @ green
        t2_green = 2 * t2_green_c - t2_green_e
        t2g = 2 * t2g_c - t2g_e
        gt2g = oe.contract("qu,qu->", t2g, green_occ, backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2

        # two body energy
        # ref
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = oe.contract("gpj,qj->gpq", rot_chol, green, backend="jax")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * t1g
        lt1g = oe.contract("gij,ij->g", chol, t1_green, backend="jax")
        e2_1_2 = -2 * (lt1g @ lg)
        t1g1 = t1 @ green_occ.T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg1, lg1, t1g1, backend="jax")
        lt1g = oe.contract("gip,qi->gpq", ham_data["lt1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lt1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        e2_2_2_1 = -lt2g @ lg

        def scanned_fun(carry, x):
            chol_i, rot_chol_i = x
            gl_i = oe.contract("pj,ji->pi", green, chol_i, backend="jax")
            lt2_green_i = oe.contract(
                "pi,ji->pj", rot_chol_i, t2_green, backend="jax"
            )
            carry[0] += 0.5 * oe.contract(
                "pi,pi->", gl_i, lt2_green_i, backend="jax"
            )
            glgp_i = oe.contract("pi,it->pt", gl_i, greenp, backend="jax")
            l2t2_1 = oe.contract(
                "pt,qu,ptqu->",
                glgp_i,
                glgp_i,
                t2, backend="jax"
            )
            l2t2_2 = oe.contract(
                "pu,qt,ptqu->",
                glgp_i,
                glgp_i,
                t2, backend="jax"
            )
            carry[1] += 2 * l2t2_1 - l2t2_2
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol, rot_chol))
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)

        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e0 = h0 + e1_0 + e2_0 # h0 + <psi|(h1+h2)|phi>/<psi|phi>
        e1 = e1_1 + e1_2 + e2_1 + e2_2 # <psi|(t1+t2)(h1+h2)|phi>/<psi|phi>

        t = 2 * t1g + gt2g # <psi|(t1+t2)|phi>/<psi|phi>

        return jnp.real(t), jnp.real(e0), jnp.real(e1)
    
    @partial(jit, static_argnums=(0)) 
    def calc_energy_pt(self, walkers:jax.Array, ham_data: dict, wave_data: dict):
        
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch
        
        def scan_batch(carry, walker_batch):
            t, e0, e1 \
                = vmap(self._calc_energy_pt, in_axes=(0, None, None))(
                    walker_batch, ham_data, wave_data)
            return carry, (t, e0, e1)
        
        _, (t, e0, e1) \
            = lax.scan(
                scan_batch, None, 
                walkers.reshape(self.n_batch, batch_size, self.norb, self.nelec[0]))
        
        t = t.reshape(n_walkers)
        e0 = e0.reshape(n_walkers)
        e1 = e1.reshape(n_walkers)
        
        return t, e0, e1


    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        
        # nocc = wave_data['t1'].shape[0]
        norb = self.norb
        # wave_data["mo_coeff"] = np.eye(norb)[:,:nocc]
        # t1, t2 = wave_data['t1'],wave_data['t2']

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
            ham_data["chol"].reshape(-1, norb, norb), backend="jax"
        )

        ham_data["lt1"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["t1"],
            optimize="optimal", backend="jax"
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    
@dataclass
class ccsd_pt2(rhf):
    """Tensor contraction form of the UCCSD_PT2 (exact T1) trial wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def thouless_trans(self, t1):
        ''' thouless transformation |psi'> = exp(t1)|psi>
            gives the transformed mo_occrep in the 
            original mo basis <psi_p|psi'_i>
            t = t_ia
            t_ia = c_ik c.T_ka
            c_ik = <psi_i|psi'_k>
        '''
        q, r = jnp.linalg.qr(t1,mode='complete')
        u_ji = q
        u_ai = r.T
        mo_t = jnp.vstack((u_ji,u_ai))
        mo_t, _ = jnp.linalg.qr(mo_t)# ,mode='complete')
        # this sgn is a problem when
        # turn on mol point group symmetry
        # sgn = jnp.sign((mo_t).diagonal())
        # choose the mo_t s.t it has positive olp with the original mo
        # <psi'_i|psi_i>
        # mo_t = jnp.einsum("ij,j->ij", mo_t, sgn)
        return mo_t
    
    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ):
        nocc = self.nelec[0]
        mo_t, t2 = wave_data["mo_t"], wave_data["t2"]
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        green = (walker @ (jnp.linalg.inv(mo_t.T @ walker)) @ mo_t.T).T
        greenp = (green - jnp.eye(self.norb))[:,nocc:]

        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1_0 = 2 * hg

        # double excitations
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

        # two body energy
        lg = oe.contract("gpq,pq->g", chol, green, backend="jax")

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
            l2t2_1 = oe.contract("ia,jb,iajb->",glgp_i,glgp_i,t2,backend="jax")
            l2t2_2 = oe.contract("ib,ja,iajb->",glgp_i,glgp_i,t2,backend="jax")
            carry[2] += 2 * l2t2_1 - l2t2_2
            return carry, 0.0

        [e2_0, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0,0.0,0.0], chol)
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        # <HF|walker>
        o0 = jnp.linalg.det(walker[:nocc,:nocc]) ** 2
        # <exp(T1)HF|walker>/<HF|walker>
        t1 = jnp.linalg.det(wave_data["mo_t"].T.conj() @ walker)**2 / o0
        t2 = gt2g * t1 # <exp(T1)HF|T2|walker>/<HF|walker>
        e0 = (e1_0 + e2_0) * t1 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
        e1 = (e1_2 + e2_2) * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>

        return jnp.real(t1), jnp.real(t2), jnp.real(e0), jnp.real(e1)

    # @singledispatchmethod
    # def calc_energy_pt(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
    #     raise NotImplementedError("Walker type not supported")

    @partial(jit, static_argnums=0)
    def calc_energy_pt(self,walkers,ham_data,wave_data):
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return t1, t2, e0, e1
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

@dataclass
class ccsd_pt_ad(rhf):
    """differential form of the CCSD_PT wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t1t2_walker_olp(self,walker,wave_data):
        ''' <psi_0(t1+t2)|phi> '''
        t1, t2 = wave_data['t1'], wave_data['t2']
        nocc = walker.shape[1]
        # GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
        GF = self._calc_green(walker,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        o1 = oe.contract("ia,ia", t1, GF[:, nocc:], backend="jax")
        o2 = 2 * oe.contract("iajb, ia, jb", t2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", t2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return (2*o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _t1t2_olp_exp1(self, x: float, h1_mod: jax.Array, walker: jax.Array,
                    wave_data: dict) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h1_mod)|walker>
        '''
        t = x * h1_mod
        walker_1x = walker + t.dot(walker)
        olp = self._t1t2_walker_olp(walker_1x,wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _t1t2_olp_exp2(self, x: float, chol_i: jax.Array, walker: jax.Array,
                    wave_data: dict) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h2_mod)|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        olp = self._t1t2_walker_olp(walker_2x,wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker, ham_data, wave_data):
        ''' <psi_0|(t1+t2)(h1+h2)|phi>/<psi_0|phi> '''

        eps=3e-4

        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(-1, norb, norb)

        # one body
        x = 0.0
        f1 = lambda a: self._t1t2_olp_exp1(a,h1_mod,walker,wave_data)
        olp_t, d_olp = jvp(f1, [x], [1.0])

        # two body
        # c_ij^ab <psi_ij^ab|phi_2x>/<psi_0|phi>
        def scanned_fun(carry, c):
            eps, walker, wave_data = carry
            return carry, self._t1t2_olp_exp2(eps,c,walker,wave_data)

        _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps
        
        o0 = self._calc_overlap_restricted(walker, wave_data)
        t = olp_t/o0
        e0 = self._calc_energy_restricted(walker,ham_data,wave_data)
        e1 = (d_olp + jnp.sum(d_2_olp) / 2.0 ) / o0

        return jnp.real(t), jnp.real(e0), jnp.real(e1)

    @partial(jit, static_argnums=(0)) 
    def calc_energy_pt(self,walkers,ham_data,wave_data):
        t, e0, e1 = vmap(
            self._calc_energy_pt,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return t, e0, e1
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""

        norb = self.norb

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
            ham_data["chol"].reshape(-1, norb, norb), backend="jax"
        )

        chol = ham_data["chol"].reshape(-1, norb, norb)
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        v0 = 0.5 * oe.contract("gik,gjk->ij",
                                chol.reshape(-1, norb, norb),
                                chol.reshape(-1, norb, norb),
                                backend="jax")
        h1_mod = h1 - v0 
        ham_data['h1_mod'] = h1_mod
        return ham_data
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt2_ad(rhf):

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def thouless_trans(self, t1):
        ''' thouless transformation |psi'> = exp(t1)|psi>
            gives the transformed mo_occrep in the 
            original mo basis <psi_p|psi'_i>
            t = t_ia
            t_ia = c_ik c.T_ka
            c_ik = <psi_i|psi'_k>
        '''
        q, r = jnp.linalg.qr(t1,mode='complete')
        u_ji = q
        u_ai = r.T
        u_occ = jnp.vstack((u_ji,u_ai))
        mo_t, _ = jnp.linalg.qr(u_occ,mode='complete')
        return mo_t

    @partial(jit, static_argnums=0)
    def _tls_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        '''<exp(T1)HF|a_p^dagger a_q|walker>/<exp(T1)HF|walker>'''
        tls_gf = (walker.dot(
                jnp.linalg.inv(wave_data["mo_t"].T.conj() @ walker))
                ).T
        return tls_gf

    @partial(jit, static_argnums=0)
    def _tls_walker_olp(self, walker, wave_data):
        ''' 
        <exp(T1)HF|walker>
        '''
        o_t = jnp.linalg.det(wave_data["mo_t"].T.conj() @ walker) ** 2
        return o_t

    @partial(jit, static_argnums=0)
    def _tls_exp1(self, x, h1_mod, walker, wave_data) -> complex:
        '''
        <exp(T1)HF|exp(x*h1_mod)|walker>/<HF|walker>
        '''
        t = x * h1_mod
        walker_1x = walker + t.dot(walker)
        t1 = self._tls_walker_olp(walker_1x,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        return t1/o0

    @partial(jit, static_argnums=0)
    def _tls_exp2(self, x, chol_i, walker, wave_data) -> complex:
        '''
        <exp(T1)HF|exp(x*h2_mod)|walker>/<HF|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        t2 = self._tls_walker_olp(walker_2x,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        return t2/o0

    @partial(jit, static_argnums=0)
    def _t2_tls_walker_olp(self, walker, wave_data):
        ''' 
        <exp(T1)HF|T2|walker>
        = t_iajb <exp(T1)HF|ijab|phi>/<exp(T1)HF|phi> * <exp(T1)HF|phi>
        '''
        rot_t2 = wave_data['rot_t2']
        nocc = walker.shape[1]
        GF = self._tls_green(walker, wave_data)
        o_t = self._tls_walker_olp(walker, wave_data)
        t2 = 2 * oe.contract(
            "iajb, ia, jb", rot_t2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", rot_t2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return t2 * o_t

    @partial(jit, static_argnums=0)
    def _t2_tls_exp1(self, x, h1_mod, walker, wave_data) -> complex:
        '''
        <exp(T1)HF|T2 exp(x*h1_mod)|walker>/<HF|walker>
        '''
        t = x * h1_mod
        walker_1x = walker + t.dot(walker)
        t2 = self._t2_tls_walker_olp(walker_1x,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        return t2/o0

    @partial(jit, static_argnums=0)
    def _t2_tls_exp2(self, x, chol_i, walker, wave_data) -> complex:
        '''
        t_iajb <psi|ijab exp(x*h2_mod)|walker>/<psi_0|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        t2 = self._t2_tls_walker_olp(walker_2x,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        return t2/o0

    @partial(jit, static_argnums=0)
    def _calc_energy_pt_restricted(self, walker, ham_data, wave_data):
        ''' 
        t1 = <exp(T1)HF|walker>/<HF|walker>
        t2 = <exp(T1)HF|T2|walker>/<HF|walker>
        e0 = <exp(T1)HF|H|walker>/<HF|walker>
        e1 = <exp(T1)HF|T2(h1+h2)|walker>/<HF|walker>
        '''

        eps = 1e-4
        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(-1, norb, norb)

        # <exp(T1)HF|h1+h2|walker>/<HF|walker>
        # one body
        # <exp(T1)HF|walker_1x>/<HF|walker>
        x = 0.0
        f1 = lambda a: self._tls_exp1(a,h1_mod,walker,wave_data)
        t1, d_exp1 = jvp(f1, [x], [1.0])

        # two body
        # <exp(T1)HF|walker_2x>/<HF|walker>
        def scanned_fun(carry, c):
            eps, walker, wave_data = carry
            return carry, self._tls_exp2(eps,c,walker,wave_data)

        _, exp2_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, exp2_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, exp2_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d2_exp2 = (exp2_p - 2.0 * exp2_0 + exp2_m) / eps / eps

        e0 = (d_exp1 + jnp.sum(d2_exp2) / 2.0 )

        d_exp1, d2_exp2 = None, None
        exp2_p, exp2_0, exp2_m = None, None, None

        # <exp(T1)HF|T2(h1+h2)|walker>/<HF|walker>
        # one body
        # <exp(T1)HF|T2|walker_1x>/<HF|walker>
        x = 0.0
        f1 = lambda a: self._t2_tls_exp1(a,h1_mod,walker,wave_data)
        t2, d_exp1 = jvp(f1, [x], [1.0])

        # two body
        # <exp(T1)HF|T2|walker_2x>/<HF|walker>
        def scanned_fun(carry, c):
            eps, walker, wave_data = carry
            return carry, self._t2_tls_exp2(eps,c,walker,wave_data)

        _, exp2_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, exp2_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, exp2_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d2_exp2 = (exp2_p - 2.0 * exp2_0 + exp2_m) / eps / eps
    
        e1 = (d_exp1 + jnp.sum(d2_exp2) / 2.0 )

        return jnp.real(t1), jnp.real(t2), jnp.real(e0), jnp.real(e1)
    
    @partial(jit, static_argnums=0)
    def calc_energy_pt(self,walkers,ham_data,wave_data):
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt_restricted,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return t1, t2, e0, e1

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class uccsd_pt(uhf):
    """A manual implementation of the UCCSD_PT wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a, t1_a, t2_aa = self.nelec[0], wave_data["t1a"], wave_data["t2aa"]
        nocc_b, t1_b, t2_bb = self.nelec[1], wave_data["t1b"], wave_data["t2bb"]
        t2_ab = wave_data["t2ab"]
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]
        h1_a = ham_data["h1"][0]
        h1_b = ham_data["h1"][1]
        hg_a = oe.contract("pj,pj->", h1_a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1_b[:nocc_b, :], green_b, backend="jax")
        hg = hg_a + hg_b

        # 0 body energy
        h0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = hg # <HF|h1|walker>/<HF|walker>

        # single excitations
        t1g_a = oe.contract("ia,ia->", t1_a, green_occ_a, backend="jax")
        t1g_b = oe.contract("ia,ia->", t1_b, green_occ_b, backend="jax")
        t1g = t1g_a + t1g_b
        e1_1_1 = t1g * hg
        gpt1_a = greenp_a @ t1_a.T
        gpt1_b = greenp_b @ t1_b.T
        t1_green_a = gpt1_a @ green_a
        t1_green_b = gpt1_b @ green_b
        e1_1_2 = -(
            oe.contract("pq,pq->", h1_a, t1_green_a, backend="jax")
            + oe.contract("pq,pq->", h1_b, t1_green_b, backend="jax")
        )
        e1_1 = e1_1_1 + e1_1_2 # <HF|T1 h1|walker>/<HF|walker>

        # double excitations
        t2g_a = oe.contract("ptqu,pt->qu", t2_aa, green_occ_a, backend="jax") / 4
        t2g_b = oe.contract("ptqu,pt->qu", t2_bb, green_occ_b, backend="jax") / 4
        t2g_ab_a = oe.contract("ptqu,qu->pt", t2_ab, green_occ_b, backend="jax")
        t2g_ab_b = oe.contract("ptqu,pt->qu", t2_ab, green_occ_a, backend="jax")
        gt2g_a = oe.contract("qu,qu->", t2g_a, green_occ_a, backend="jax")
        gt2g_b = oe.contract("qu,qu->", t2g_b, green_occ_b, backend="jax")
        gt2g_ab = oe.contract("pt,pt->", t2g_ab_a, green_occ_a, backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab
        e1_2_1 = hg * gt2g
        t2_green_a = (greenp_a @ t2g_a.T) @ green_a
        t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_a
        t2_green_b = (greenp_b @ t2g_b.T) @ green_b
        t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_b
        e1_2_2_a = -oe.contract(
            "ij,ij->", h1_a, 4 * t2_green_a + t2_green_ab_a, backend="jax"
        )
        e1_2_2_b = -oe.contract(
            "ij,ij->", h1_b, 4 * t2_green_b + t2_green_ab_b, backend="jax"
        )
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2 # <HF|T2 h1|walker>/<HF|walker>

        # two body energy
        # ref
        lg_a = oe.contract("gpj,pj->g", rot_chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpj,pj->g", rot_chol_b, green_b, backend="jax")
        e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
        lg1_a = oe.contract("gpj,qj->gpq", rot_chol_a, green_a, backend="jax")
        lg1_b = oe.contract("gpj,qj->gpq", rot_chol_b, green_b, backend="jax")
        e2_0_2 = (
            -(
                jnp.sum(vmap(lambda x: x * x.T)(lg1_a))
                + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))
            )
            / 2.0
        )
        e2_0 = e2_0_1 + e2_0_2 # <HF|h2|walker>/<HF|walker>

        # single excitations
        e2_1_1 = e2_0 * t1g
        lt1g_a = oe.contract("gij,ij->g", chol_a, t1_green_a, backend="jax")
        lt1g_b = oe.contract("gij,ij->g", chol_b, t1_green_b, backend="jax")
        e2_1_2 = -((lt1g_a + lt1g_b) @ (lg_a + lg_b))
        t1g1_a = t1_a @ green_occ_a.T
        t1g1_b = t1_b @ green_occ_b.T
        e2_1_3_1 = oe.contract(
            "gpq,gqr,rp->", lg1_a, lg1_a, t1g1_a, backend="jax"
        ) + oe.contract("gpq,gqr,rp->", lg1_b, lg1_b, t1g1_b, backend="jax")
        lt1g_a = oe.contract(
            "gip,qi->gpq", ham_data["lt1_a"], green_a, backend="jax"
        )
        lt1g_b = oe.contract(
            "gip,qi->gpq", ham_data["lt1_b"], green_b, backend="jax"
        )
        e2_1_3_2 = -oe.contract(
            "gpq,gqp->", lt1g_a, lg1_a, backend="jax"
        ) - oe.contract("gpq,gqp->", lt1g_b, lg1_b, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + e2_1_2 + e2_1_3 # <HF|T1 h2|walker>/<HF|walker>

        # double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g_a = oe.contract(
            "gij,ij->g",
            chol_a,
            8 * t2_green_a + 2 * t2_green_ab_a,
            backend="jax",
        )
        lt2g_b = oe.contract(
            "gij,ij->g",
            chol_b,
            8 * t2_green_b + 2 * t2_green_ab_b,
            backend="jax",
        )
        e2_2_2_1 = -((lt2g_a + lt2g_b) @ (lg_a + lg_b)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = oe.contract("pj,ji->pi", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("pj,ji->pi", green_b, chol_b_i, backend="jax")
            lt2_green_a_i = oe.contract(
                "pi,ji->pj",
                rot_chol_a_i,
                8 * t2_green_a + 2 * t2_green_ab_a,
                backend="jax",
            )
            lt2_green_b_i = oe.contract(
                "pi,ji->pj",
                rot_chol_b_i,
                8 * t2_green_b + 2 * t2_green_ab_b,
                backend="jax",
            )
            carry[0] += 0.5 * (
                oe.contract("pi,pi->", gl_a_i, lt2_green_a_i, backend="jax")
                + oe.contract("pi,pi->", gl_b_i, lt2_green_b_i, backend="jax")
            )
            glgp_a_i = oe.contract(
                "pi,it->pt", gl_a_i, greenp_a, backend="jax"
            )
            glgp_b_i = oe.contract(
                "pi,it->pt", gl_b_i, greenp_b, backend="jax"
            )
            l2t2_a = 0.5 * oe.contract(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_a_i,
                t2_aa,
                backend="jax",
            )
            l2t2_b = 0.5 * oe.contract(
                "pt,qu,ptqu->",
                glgp_b_i,
                glgp_b_i,
                t2_bb,
                backend="jax",
            )
            l2t2_ab = oe.contract(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_b_i,
                t2_ab,
                backend="jax",
            )
            carry[1] += l2t2_a + l2t2_b + l2t2_ab
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <HF|T2 h2|walker>/<HF|walker>

        t = t1g + gt2g # <HF|T1+T2|walker>/<HF|walker>
        e0 = h0 + e1_0 + e2_0 # h0 + <HF|h1+h2|walker>/<HF|walker>
        e1 = e1_1 + e1_2 + e2_1 + e2_2 # <HF|(T1+T2)(h1+h2)|walker>/<HF|walker>

        return jnp.real(t), jnp.real(e0), jnp.real(e1)

    @singledispatchmethod
    def calc_energy_pt(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        raise NotImplementedError("Walker type not supported")

    @calc_energy_pt.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t, e0, e1
    
    @calc_energy_pt.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        t, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers, walkers, ham_data, wave_data)
        return t, e0, e1

    @partial(jit, static_argnums=0)
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
                ham_data["chol"][0].reshape(-1, self.norb, self.norb), backend="jax"
            ),
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][1].T.conj(),
                ham_data["chol"][1].reshape(-1, self.norb, self.norb), backend="jax"
            ),
        ]
        ham_data["lt1_a"] = oe.contract(
            "gpa,ia->gpi",
            ham_data["chol"][0].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["t1a"],
            backend="jax"
        )
        ham_data["lt1_b"] = oe.contract(
            "gpa,ia->gpi",
            ham_data["chol"][1].reshape(-1, self.norb, self.norb)[:, :, self.nelec[1] :],
            wave_data["t1b"],
            backend="jax"
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class uccsd_pt2(uhf):
    """Tensor contraction form of the UCCSD_PT2 (exact T1) trial wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def thouless_trans(self, t1):
        ''' thouless transformation |psi'> = exp(t1)|psi>
            gives the transformed mo_occrep in the 
            original mo basis <psi_p|psi'_i>
            t = t_ia
            t_ia = c_ik c.T_ka
            c_ik = <psi_i|psi'_k>
        '''
        q, r = jnp.linalg.qr(t1,mode='complete')
        u_ji = q
        u_ai = r.T
        mo_t = jnp.vstack((u_ji,u_ai))
        mo_t, _ = jnp.linalg.qr(mo_t)# ,mode='complete')
        # this sgn is a problem when
        # turn on mol point group symmetry
        # sgn = jnp.sign((mo_t).diagonal())
        # choose the mo_t s.t it has positive olp with the original mo
        # <psi'_i|psi_i>
        # mo_t = jnp.einsum("ij,j->ij", mo_t, sgn)
        return mo_t

    # @partial(jit, static_argnums=0)
    # def _calc_energy_pt(
    #     self,
    #     walker_up: jax.Array,
    #     walker_dn: jax.Array,
    #     ham_data: dict,
    #     wave_data: dict,
    # ) -> complex:
    #     '''
    #     t1 = <exp(T1)HF|walker>/<HF|walker>
    #     t2 = <exp(T1)HF|T2|walker>/<HF|walker>
    #     e0 = <exp(T1)HF|h1+h2|walker>/<HF|walker>
    #     e1 = <exp(T1)HF|T2(h1+h2)|walker>/<HF|walker>
    #     '''
    #     nocc_a, t2_aa = self.nelec[0], wave_data["t2aa"]
    #     nocc_b, t2_bb = self.nelec[1], wave_data["t2bb"]
    #     t2_ab = wave_data["t2ab"]
    #     mo_ta, mo_tb = wave_data['mo_ta'], wave_data['mo_tb']
    #     chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
    #     chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
    #     h1_a = ham_data["h1"][0]
    #     h1_b = ham_data["h1"][1]

    #     # full green's function G_pq
    #     green_a = (walker_up @ (jnp.linalg.inv(mo_ta.T @ walker_up)) @ mo_ta.T).T
    #     green_b = (walker_dn @ (jnp.linalg.inv(mo_tb.T @ walker_dn)) @ mo_tb.T).T
    #     # Gp_pa = G_pa - delta_pa
    #     greenp_a = (green_a - jnp.eye(self.norb))[:,nocc_a:]
    #     greenp_b = (green_b - jnp.eye(self.norb))[:,nocc_b:]

    #     hg_a = jnp.einsum("pq,pq->", h1_a, green_a)
    #     hg_b = jnp.einsum("pq,pq->", h1_b, green_b)
    #     hg = hg_a + hg_b # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>

    #     # 0 body energy
    #     # h0 = ham_data["h0"]

    #     # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>
    #     # one body energy
    #     e1_0 = hg

    #     # <exp(T1)HF|h2|walker>/<exp(T1)HF|walker>
    #     # two body energy
    #     lg_a = jnp.einsum("gpq,pq->g", chol_a, green_a, optimize="optimal")
    #     lg_b = jnp.einsum("gpq,pq->g", chol_b, green_b, optimize="optimal")
    #     e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
    #     lg1_a = jnp.einsum("gpj,qj->gpq", chol_a, green_a, optimize="optimal")
    #     lg1_b = jnp.einsum("gpj,qj->gpq", chol_b, green_b, optimize="optimal")
    #     e2_0_2 = (
    #         -(
    #             jnp.sum(vmap(lambda x: x * x.T)(lg1_a))
    #             + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))
    #         )
    #         / 2.0
    #     )
    #     e2_0 = e2_0_1 + e2_0_2

    #     # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>
    #     # double excitations
    #     t2g_a = jnp.einsum("iajb,ia->jb", t2_aa, green_a[:nocc_a,nocc_a:]) / 4
    #     t2g_b = jnp.einsum("iajb,ia->jb", t2_bb, green_b[:nocc_b,nocc_b:]) / 4
    #     t2g_ab_a = jnp.einsum("iajb,jb->ia", t2_ab, green_b[:nocc_b,nocc_b:])
    #     t2g_ab_b = jnp.einsum("iajb,ia->jb", t2_ab, green_a[:nocc_a,nocc_a:])
    #     # t_iajb (G_ia G_jb - G_ib G_ja)
    #     gt2g_a = jnp.einsum("jb,jb->", t2g_a, green_a[:nocc_a,nocc_a:], 
    #                         optimize="optimal")
    #     gt2g_b = jnp.einsum("jb,jb->", t2g_b, green_b[:nocc_b,nocc_b:], 
    #                         optimize="optimal")
    #     gt2g_ab = jnp.einsum("ia,ia->", t2g_ab_a, green_a[:nocc_a,nocc_a:], 
    #                         optimize="optimal")
    #     gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>

    #     e1_2_1 = hg * gt2g
        
    #     t2_green_a = (greenp_a @ t2g_a.T) @ green_a[:nocc_a,:] # Gp_pb t_iajb G_ia G_jq
    #     t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a,:]
    #     t2_green_b = (greenp_b @ t2g_b.T) @ green_b[:nocc_b,:]
    #     t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b,:]
    #     e1_2_2_a = -jnp.einsum(
    #         "pq,pq->", h1_a, 4 * t2_green_a + t2_green_ab_a, optimize="optimal"
    #     )
    #     e1_2_2_b = -jnp.einsum(
    #         "pq,pq->", h1_b, 4 * t2_green_b + t2_green_ab_b, optimize="optimal"
    #     )
    #     e1_2_2 = e1_2_2_a + e1_2_2_b
    #     e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

    #     # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>
    #     # double excitations
    #     e2_2_1 = e2_0 * gt2g
    #     lt2g_a = jnp.einsum("gpq,pq->g",
    #                         chol_a, 8 * t2_green_a + 2 * t2_green_ab_a,
    #                         optimize="optimal")
    #     lt2g_b = jnp.einsum("gpq,pq->g",
    #         chol_b, 8 * t2_green_b + 2 * t2_green_ab_b,
    #         optimize="optimal")
    #     e2_2_2_1 = -((lt2g_a + lt2g_b) @ (lg_a + lg_b)) / 2.0

    #     def scanned_fun(carry, x):
    #         chol_a_i, chol_b_i = x
    #         gl_a_i = jnp.einsum("pr,rq->pq", green_a, chol_a_i,
    #                             optimize="optimal")
    #         gl_b_i = jnp.einsum("pr,rq->pq", green_b, chol_b_i,
    #                             optimize="optimal")
    #         lt2_green_a_i = jnp.einsum(
    #             "pr,qr->pq", chol_a_i, 8 * t2_green_a + 2 * t2_green_ab_a,
    #             optimize="optimal")
    #         lt2_green_b_i = jnp.einsum(
    #             "pr,qr->pq", chol_b_i, 8 * t2_green_b + 2 * t2_green_ab_b,
    #             optimize="optimal")
    #         carry[0] += 0.5 * (
    #             jnp.einsum("pq,pq->", gl_a_i, lt2_green_a_i, optimize="optimal")
    #             + jnp.einsum("pq,pq->", gl_b_i, lt2_green_b_i, optimize="optimal")
    #         )
    #         glgp_a_i = jnp.einsum(
    #             "iq,qa->ia", gl_a_i[:nocc_a,:], greenp_a, optimize="optimal"
    #         ) #.astype(jnp.complex64)
    #         glgp_b_i = jnp.einsum(
    #             "iq,qa->ia", gl_b_i[:nocc_b,:], greenp_b, optimize="optimal"
    #         ) #.astype(jnp.complex64)
    #         l2t2_a = 0.5 * jnp.einsum(
    #             "ia,jb,iajb->",glgp_a_i,glgp_a_i,t2_aa, #.astype(jnp.float32),
    #             optimize="optimal")
    #         l2t2_b = 0.5 * jnp.einsum(
    #             "ia,jb,iajb->",glgp_b_i,glgp_b_i,t2_bb, #.astype(jnp.float32),
    #             optimize="optimal")
    #         l2t2_ab = jnp.einsum(
    #             "ia,jb,iajb->",glgp_a_i,glgp_b_i,t2_ab, #.astype(jnp.float32),
    #             optimize="optimal")
    #         carry[1] += l2t2_a + l2t2_b + l2t2_ab
    #         return carry, 0.0

    #     [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol_a, chol_b))
    #     e2_2_2 = e2_2_2_1 + e2_2_2_2
    #     e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>

    #     # e2 = e2_0 + e2_1 + e2_2
    #     o0 = jnp.linalg.det(walker_up[:nocc_a,:nocc_a]
    #         ) * jnp.linalg.det(walker_dn[:nocc_b,:nocc_b])
    #     # <exp(T1)HF|walker>/<HF|walker>
    #     t1 = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
    #         ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn) / o0
    #     t2 = gt2g * t1 # <exp(T1)HF|T2|walker>/<HF|walker>
    #     e0 = (e1_0 + e2_0) * t1 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
    #     e1 = (e1_2 + e2_2) * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>

    #     return jnp.real(t1), jnp.real(t2), jnp.real(e0), jnp.real(e1)
    
    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a, t2_aa = self.nelec[0], wave_data["t2aa"]
        nocc_b, t2_bb = self.nelec[1], wave_data["t2bb"]
        t2_ab = wave_data["t2ab"]
        mo_a, mo_b = wave_data['mo_ta'], wave_data['mo_tb']
        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        h1_a = ham_data["h1"][0]
        h1_b = ham_data["h1"][1]

        # full green's function G_pq
        green_a = (walker_up @ (jnp.linalg.inv(mo_a.T @ walker_up)) @ mo_a.T).T
        green_b = (walker_dn @ (jnp.linalg.inv(mo_b.T @ walker_dn)) @ mo_b.T).T
        greenp_a = (green_a - jnp.eye(self.norb))[:,nocc_a:]
        greenp_b = (green_b - jnp.eye(self.norb))[:,nocc_b:]

        hg_a = oe.contract("pq,pq->", h1_a, green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b, green_b, backend="jax")
        hg = hg_a + hg_b # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>

        # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>
        # one body energy
        e1_0 = hg

        # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>
        # double excitations
        t2g_a = oe.contract("iajb,ia->jb", t2_aa, green_a[:nocc_a,nocc_a:],
                            backend="jax") / 4
        t2g_b = oe.contract("iajb,ia->jb", t2_bb, green_b[:nocc_b,nocc_b:], 
                            backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,jb->ia", t2_ab, green_b[:nocc_b,nocc_b:],
                               backend="jax")
        t2g_ab_b = oe.contract("iajb,ia->jb", t2_ab, green_a[:nocc_a,nocc_a:],
                               backend="jax")
        # t_iajb (G_ia G_jb - G_ib G_ja)
        gt2g_a = oe.contract("jb,jb->", t2g_a, green_a[:nocc_a,nocc_a:], 
                            backend="jax")
        gt2g_b = oe.contract("jb,jb->", t2g_b, green_b[:nocc_b,nocc_b:], 
                            backend="jax")
        gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, green_a[:nocc_a,nocc_a:], 
                              backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>

        e1_2_1 = hg * gt2g
        
        t2_green_a = (greenp_a @ t2g_a.T) @ green_a[:nocc_a,:] # Gp_pb t_iajb G_ia G_jq
        t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a,:]
        t2_green_b = (greenp_b @ t2g_b.T) @ green_b[:nocc_b,:]
        t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b,:]
        e1_2_2_a = -oe.contract(
            "pq,pq->", h1_a, 4 * t2_green_a + t2_green_ab_a, backend="jax")
        e1_2_2_b = -oe.contract(
            "pq,pq->", h1_b, 4 * t2_green_b + t2_green_ab_b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

        # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>
        # double excitations
        # e2_2_1 = e2_0 * gt2g
        lg_a = oe.contract("gpq,pq->g", chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpq,pq->g", chol_b, green_b, backend="jax")
        lt2g_a = oe.contract("gpq,pq->g",
                            chol_a, 8 * t2_green_a + 2 * t2_green_ab_a,
                            backend="jax")
        lt2g_b = oe.contract("gpq,pq->g",
            chol_b, 8 * t2_green_b + 2 * t2_green_ab_b,
            backend="jax")
        e2_2_2_1 = -((lt2g_a + lt2g_b) @ (lg_a + lg_b)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, chol_b_i = x
            # e2_0
            lg_a_i = oe.contract("pr,qr->pq", chol_a_i, green_a, backend="jax")
            lg_b_i = oe.contract("pr,qr->pq", chol_b_i, green_b, backend="jax")
            e2_0_1_i = (jnp.trace(lg_a_i) + jnp.trace(lg_b_i))**2 / 2.0
            e2_0_2_i = -(oe.contract('pq,qp->',lg_a_i,lg_a_i, backend="jax") 
                        + oe.contract('pq,qp->',lg_b_i,lg_b_i, backend="jax")
                        ) / 2.0
            carry[0] += e2_0_1_i + e2_0_2_i
            # e2_2
            gl_a_i = oe.contract("pr,rq->pq", green_a, chol_a_i,
                                backend="jax")
            gl_b_i = oe.contract("pr,rq->pq", green_b, chol_b_i,
                                backend="jax")
            lt2_green_a_i = oe.contract(
                "pr,qr->pq", chol_a_i, 8 * t2_green_a + 2 * t2_green_ab_a,
                backend="jax")
            lt2_green_b_i = oe.contract(
                "pr,qr->pq", chol_b_i, 8 * t2_green_b + 2 * t2_green_ab_b,
                backend="jax")
            carry[1] += 0.5 * (
                oe.contract("pq,pq->", gl_a_i, lt2_green_a_i, backend="jax")
                + oe.contract("pq,pq->", gl_b_i, lt2_green_b_i, backend="jax")
            )
            glgp_a_i = oe.contract(
                "iq,qa->ia", gl_a_i[:nocc_a,:], greenp_a, backend="jax"
            )
            glgp_b_i = oe.contract(
                "iq,qa->ia", gl_b_i[:nocc_b,:], greenp_b, backend="jax"
            )
            l2t2_a = 0.5 * oe.contract(
                "ia,jb,iajb->",glgp_a_i,glgp_a_i,t2_aa,
                backend="jax")
            l2t2_b = 0.5 * oe.contract(
                "ia,jb,iajb->",glgp_b_i,glgp_b_i,t2_bb,
                backend="jax")
            l2t2_ab = oe.contract(
                "ia,jb,iajb->",glgp_a_i,glgp_b_i,t2_ab,
                backend="jax")
            carry[2] += l2t2_a + l2t2_b + l2t2_ab
            return carry, 0.0

        [e2_0, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0, 0.0], (chol_a, chol_b))
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>

        o0 = jnp.linalg.det(walker_up[:nocc_a,:nocc_a]
            ) * jnp.linalg.det(walker_dn[:nocc_b,:nocc_b])
        # <exp(T1)HF|walker>/<HF|walker>
        t1 = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
            ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn) / o0
        t2 = gt2g * t1 # <exp(T1)HF|T2|walker>/<HF|walker>
        e0 = (e1_0 + e2_0) * t1 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
        e1 = (e1_2 + e2_2) * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>

        return jnp.real(t1), jnp.real(t2), jnp.real(e0), jnp.real(e1)

    @singledispatchmethod
    def calc_energy_pt(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        raise NotImplementedError("Walker type not supported")

    @calc_energy_pt.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t1, t2, e0, e1
    
    @calc_energy_pt.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt2, in_axes=(0, 0, None, None))(
            walkers, walkers, ham_data, wave_data)
        return t1, t2, e0, e1


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class uccsd_pt_ad(uhf):
    """differential form of the CCSD_PT wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t1t2_walker_olp(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        '''<HF|(t1+t2)|walker> = (t_ia G_ia + t_iajb G_iajb) * <HF|walker>'''
        noccA, t1A, t2AA = self.nelec[0], wave_data["rot_t1A"], wave_data["rot_t2AA"]
        noccB, t1B, t2BB = self.nelec[1], wave_data["rot_t1B"], wave_data["rot_t2BB"]
        t2AB = wave_data["rot_t2AB"]
        # green_a = (walker_up.dot(jnp.linalg.inv(wave_data["mo_coeff"][0].T.conj() @ walker_up))).T
        # green_b = (walker_dn.dot(jnp.linalg.inv(wave_data["mo_coeff"][1].T.conj() @ walker_dn))).T
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:noccA,:noccA]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:noccB,:noccB]))).T
        green_a, green_b = green_a[:noccA, noccA:], green_b[:noccB, noccB:]
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        o1 = oe.contract("ia,ia", t1A, green_a, backend="jax") \
              + oe.contract("ia,ia", t1B, green_b, backend="jax")
        o2 = (
            0.5 * oe.contract("iajb, ia, jb", t2AA, green_a, green_a, backend="jax")
            + 0.5 * oe.contract("iajb, ia, jb", t2BB, green_b, green_b, backend="jax")
            + oe.contract("iajb, ia, jb", t2AB, green_a, green_b, backend="jax")
        )
        return (o1 + o2) * o0
    
    @partial(jit, static_argnums=0)
    def _t1t2_exp1(self, x: float, h1_mod: jax.Array, walker_up: jax.Array,
                        walker_dn: jax.Array, wave_data: dict):
        '''
        unrestricted t_ia <psi_i^a|exp(x*h1_mod)|walker>/<HF|walker>
        '''
        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)
        
        olp = self._t1t2_walker_olp(walker_up_1x, walker_dn_1x, wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _t1t2_exp2(self, x: float, chol_i: jax.Array,
                   walker_up: jax.Array, walker_dn: jax.Array,
                   wave_data: dict) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h2_mod)|walker>/<HF|walker>
        '''

        walker_up_2x = (
            walker_up
            + x * chol_i[0].dot(walker_up)
            + x**2 / 2.0 * chol_i[0].dot(chol_i[0].dot(walker_up))
        )
        walker_dn_2x = (
            walker_dn
            + x * chol_i[1].dot(walker_dn)
            + x**2 / 2.0 * chol_i[1].dot(chol_i[1].dot(walker_dn))
        )
        
        olp = self._t1t2_walker_olp(walker_up_2x,walker_dn_2x,wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        
        return olp/o0

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        t = <psi|T1+T2|phi>/<psi|phi>
        e0 = <psi|H|phi>/<psi|phi>
        e1 = <psi|(T1+T2)(h1+h2)|phi>/<psi|phi>
        '''

        eps=1e-4

        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)

        # one body
        x = 0.0
        f1 = lambda a: self._t1t2_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        t, d_exp1 = jvp(f1, [x], [1.0])

        # two body
        def scanned_fun(carry, c):
            eps,walker_up,walker_dn,wave_data = carry
            return carry, self._t1t2_exp2(eps,c,walker_up,walker_dn,wave_data)

        _, exp2_p = lax.scan(scanned_fun, (eps,walker_up,walker_dn,wave_data), chol)
        _, exp2_0 = lax.scan(scanned_fun, (0.0,walker_up,walker_dn,wave_data), chol)
        _, exp2_m = lax.scan(scanned_fun, (-1.0*eps,walker_up,walker_dn,wave_data), chol)
        d2_exp2 = (exp2_p - 2.0 * exp2_0 + exp2_m) / eps / eps

        e0 = self._calc_energy(walker_up,walker_dn,ham_data,wave_data)
        e1 = (d_exp1 + jnp.sum(d2_exp2) / 2.0 )

        return jnp.real(t), jnp.real(e0), jnp.real(e1)
    
    @singledispatchmethod
    def calc_energy_pt(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        raise NotImplementedError("Walker type not supported")

    @calc_energy_pt.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t, e0, e1
    
    @calc_energy_pt.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        t, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers, walkers, ham_data, wave_data)
        return t, e0, e1

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    
@dataclass
class uccsd_pt2_ad(uhf):
    """differential form of the CCSD_PT2 (exact T1) wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def thouless_trans(self, t1):
        ''' thouless transformation |psi'> = exp(t1)|psi>
            gives the transformed mo_occrep in the 
            original mo basis <psi_p|psi'_i>
            t = t_ia
            t_ia = c_ik c.T_ka
            c_ik = <psi_i|psi'_k>
        '''
        q, r = jnp.linalg.qr(t1,mode='complete')
        u_ji = q
        u_ai = r.T
        mo_t = jnp.vstack((u_ji,u_ai))
        mo_t, _ = jnp.linalg.qr(mo_t)# ,mode='complete')
        # this sgn is a problem when
        # turn on mol point group symmetry
        # sgn = jnp.sign((mo_t).diagonal())
        # choose the mo_t s.t it has positive olp with the original mo
        # <psi'_i|psi_i>
        # mo_t = jnp.einsum("ij,j->ij", mo_t, sgn)
        return mo_t
    
    @partial(jit, static_argnums=0)
    def _tls_olp(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        '''<exp(T1)HF|walker>'''

        olp = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
            ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn)

        return olp

    @partial(jit, static_argnums=0)
    def _tls_exp1(self, x: float, h1_mod: jax.Array, walker_up: jax.Array,
                        walker_dn: jax.Array, wave_data: dict):
        '''
        unrestricted <ep(T1)HF|exp(x*h1_mod)|walker>
        '''

        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)

        olp = self._tls_olp(walker_up_1x, walker_dn_1x, wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _tls_exp2(self, x: float, chol_i: jax.Array, walker_up: jax.Array,
                    walker_dn: jax.Array, wave_data: dict) -> complex:
        '''
        <exp(T1)HF|exp(x*h2_mod)|walker>
        '''

        walker_up_2x = (
            walker_up
            + x * chol_i[0].dot(walker_up)
            + x**2 / 2.0 * chol_i[0].dot(chol_i[0].dot(walker_up))
        )
        walker_dn_2x = (
            walker_dn
            + x * chol_i[1].dot(walker_dn)
            + x**2 / 2.0 * chol_i[1].dot(chol_i[1].dot(walker_dn))
        )

        olp = self._tls_olp(walker_up_2x,walker_dn_2x,wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        
        return olp/o0
    
    @partial(jit, static_argnums=0)
    def _ut2_walker_olp(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        '''<exp(T1)HF|(t1+t2)|walker> = (t_ia G_ia + t_iajb G_iajb) * <exp(T1)HF|walker>'''
        noccA, t2AA = self.nelec[0], wave_data["rot_t2AA"]
        noccB, t2BB = self.nelec[1], wave_data["rot_t2BB"]
        t2AB = wave_data["rot_t2AB"]
        mo_A = wave_data['mo_ta'] # in alpha basis
        mo_B = wave_data['mo_tb'] # in beta basis
        green_a = (walker_up.dot(jnp.linalg.inv(mo_A.T.conj() @ walker_up))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(mo_B.T.conj() @ walker_dn))).T
        green_a, green_b = green_a[:noccA, noccA:], green_b[:noccB, noccB:]
        o0 = self._tls_olp(walker_up,walker_dn,wave_data)
        o2 = (0.5 * oe.contract("iajb, ia, jb", t2AA, green_a, green_a, backend="jax")
            + 0.5 * oe.contract("iajb, ia, jb", t2BB, green_b, green_b, backend="jax")
            + oe.contract("iajb, ia, jb", t2AB, green_a, green_b, backend="jax"))
        return o2 * o0

    @partial(jit, static_argnums=0)
    def _ut2_exp1(self, x: float, h1_mod: jax.Array, walker_up: jax.Array,
                  walker_dn: jax.Array, wave_data: dict):
        '''
        unrestricted <ep(T1)HF|T2 exp(x*h1_mod)|walker>
        '''
        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)
        
        olp = self._ut2_walker_olp(walker_up_1x, walker_dn_1x, wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _ut2_exp2(self, x: float, chol_i: jax.Array, walker_up: jax.Array,
                  walker_dn: jax.Array, wave_data: dict) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h2_mod)|walker>
        '''

        walker_up_2x = (
            walker_up
            + x * chol_i[0].dot(walker_up)
            + x**2 / 2.0 * chol_i[0].dot(chol_i[0].dot(walker_up))
        )
        walker_dn_2x = (
            walker_dn
            + x * chol_i[1].dot(walker_dn)
            + x**2 / 2.0 * chol_i[1].dot(chol_i[1].dot(walker_dn))
        )
        
        olp = self._ut2_walker_olp(walker_up_2x,walker_dn_2x,wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        t1 = <exp(T1)HF|walker>/<HF|walker>
        t2 = <exp(T1)HF|T2|walker>/<HF|walker>
        e0 = <exp(T1)HF|h1+h2|walker>/<HF|walker>
        e1 = <exp(T1)HF|T2(h1+h2)|walker>/<HF|walker>
        '''

        eps=1e-4

        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)

        # e0 = <exp(T1)HF|h1+h2|walker>/<HF|walker> #
        # one body
        x = 0.0
        f1 = lambda a: self._tls_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        t1, d_exp1 = jvp(f1, [x], [1.0])

        # two body
        def scanned_fun(carry, c):
            eps,walker_up,walker_dn,wave_data = carry
            return carry, self._tls_exp2(eps,c,walker_up,walker_dn,wave_data)

        _, exp2_p = lax.scan(scanned_fun, (eps,walker_up,walker_dn,wave_data), chol)
        _, exp2_0 = lax.scan(scanned_fun, (0.0,walker_up,walker_dn,wave_data), chol)
        _, exp2_m = lax.scan(scanned_fun, (-1.0*eps,walker_up,walker_dn,wave_data), chol)
        d2_exp2 = (exp2_p - 2.0 * exp2_0 + exp2_m) / eps / eps

        e0 = (d_exp1 + jnp.sum(d2_exp2) / 2.0 )

        d_exp1 = d2_exp2 = None
        exp2_p = exp2_0 = exp2_m = None
        
        # e1 = <exp(T1)HF|(T1+T2)(h1+h2)|walker>/<HF|walker>
        # one body
        x = 0.0
        f1 = lambda a: self._ut2_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        t2, d_exp1 = jvp(f1, [x], [1.0])

        # two body
        def scanned_fun(carry, c):
            eps,walker_up,walker_dn,wave_data = carry
            return carry, self._ut2_exp2(eps,c,walker_up,walker_dn,wave_data)

        _, exp2_p = lax.scan(scanned_fun, (eps,walker_up,walker_dn,wave_data), chol)
        _, exp2_0 = lax.scan(scanned_fun, (0.0,walker_up,walker_dn,wave_data), chol)
        _, exp2_m = lax.scan(scanned_fun, (-1.0*eps,walker_up,walker_dn,wave_data), chol)
        d2_exp2 = (exp2_p - 2.0 * exp2_0 + exp2_m) / eps / eps

        e1 = (d_exp1 + jnp.sum(d2_exp2) / 2.0 )

        # o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        return jnp.real(t1), jnp.real(t2), jnp.real(e0), jnp.real(e1)

    @singledispatchmethod
    def calc_energy_pt(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        raise NotImplementedError("Walker type not supported")

    @calc_energy_pt.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t1, t2, e0, e1
    
    @calc_energy_pt.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers, walkers, ham_data, wave_data)
        return t1, t2, e0, e1

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

@dataclass
class uccsd_pt2_true_ad(uhf):
    """differential form of the CCSD_PT2 (exact T1) wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def thouless_trans(self, t1):
        ''' thouless transformation |psi'> = exp(t1)|psi>
            gives the transformed mo_occrep in the 
            original mo basis <psi_p|psi'_i>
            t = t_ia
            t_ia = c_ik c.T_ka
            c_ik = <psi_i|psi'_k>
        '''
        q, r = jnp.linalg.qr(t1,mode='complete')
        u_ji = q
        u_ai = r.T
        mo_t = jnp.vstack((u_ji,u_ai))
        mo_t, _ = jnp.linalg.qr(mo_t)# ,mode='complete')
        # this sgn is a problem when
        # turn on mol point group symmetry
        # sgn = jnp.sign((mo_t).diagonal())
        # choose the mo_t s.t it has positive olp with the original mo
        # <psi'_i|psi_i>
        # mo_t = jnp.einsum("ij,j->ij", mo_t, sgn)
        return mo_t
    
    @partial(jit, static_argnums=0)
    def _tls_olp(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        '''<exp(T1)HF|walker>'''

        olp = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
            ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn)

        return olp

    @partial(jit, static_argnums=0)
    def _tls_exp1(self, x: float, h1_mod: jax.Array, walker_up: jax.Array,
                        walker_dn: jax.Array, wave_data: dict):
        '''
        unrestricted <ep(T1)HF|exp(x*h1_mod)|walker>
        '''

        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)

        olp = self._tls_olp(walker_up_1x, walker_dn_1x, wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _tls_exp2(self, x: float, chol_i: jax.Array, walker_up: jax.Array,
                    walker_dn: jax.Array, wave_data: dict) -> complex:
        '''
        <exp(T1)HF|exp(x*h2_mod)|walker>
        '''

        walker_up_2x = (
            walker_up
            + x * chol_i[0].dot(walker_up)
            + x**2 / 2.0 * chol_i[0].dot(chol_i[0].dot(walker_up))
        )
        walker_dn_2x = (
            walker_dn
            + x * chol_i[1].dot(walker_dn)
            + x**2 / 2.0 * chol_i[1].dot(chol_i[1].dot(walker_dn))
        )

        olp = self._tls_olp(walker_up_2x,walker_dn_2x,wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        
        return olp/o0
    
    @partial(jit, static_argnums=0)
    def _ut2_walker_olp(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        '''<exp(T1)HF|(t1+t2)|walker> = (t_ia G_ia + t_iajb G_iajb) * <exp(T1)HF|walker>'''
        noccA, t2AA = self.nelec[0], wave_data["rot_t2AA"]
        noccB, t2BB = self.nelec[1], wave_data["rot_t2BB"]
        t2AB = wave_data["rot_t2AB"]
        mo_A = wave_data['mo_ta'] # in alpha basis
        mo_B = wave_data['mo_tb'] # in beta basis
        green_a = (walker_up.dot(jnp.linalg.inv(mo_A.T.conj() @ walker_up))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(mo_B.T.conj() @ walker_dn))).T
        green_a, green_b = green_a[:noccA, noccA:], green_b[:noccB, noccB:]
        o0 = self._tls_olp(walker_up,walker_dn,wave_data)
        o2 = (0.5 * oe.contract("iajb, ia, jb", t2AA, green_a, green_a, backend="jax")
            + 0.5 * oe.contract("iajb, ia, jb", t2BB, green_b, green_b, backend="jax")
            + oe.contract("iajb, ia, jb", t2AB, green_a, green_b, backend="jax"))
        return o2 * o0

    @partial(jit, static_argnums=0)
    def _ut2_exp1(self, x: float, h1_mod: jax.Array, walker_up: jax.Array,
                  walker_dn: jax.Array, wave_data: dict):
        '''
        unrestricted <ep(T1)HF|T2 exp(x*h1_mod)|walker>
        '''
        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)
        
        olp = self._ut2_walker_olp(walker_up_1x, walker_dn_1x, wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _ut2_exp2(self, x: float, chol_i: jax.Array, walker_up: jax.Array,
                  walker_dn: jax.Array, wave_data: dict) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h2_mod)|walker>
        '''

        walker_up_2x = (
            walker_up
            + x * chol_i[0].dot(walker_up)
            + x**2 / 2.0 * chol_i[0].dot(chol_i[0].dot(walker_up))
        )
        walker_dn_2x = (
            walker_dn
            + x * chol_i[1].dot(walker_dn)
            + x**2 / 2.0 * chol_i[1].dot(chol_i[1].dot(walker_dn))
        )
        
        olp = self._ut2_walker_olp(walker_up_2x,walker_dn_2x,wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0
    
    @partial(jit, static_argnums=0)
    def _d2_tls_exp2_i(self,chol_i,walker_up,walker_dn,wave_data):
        x = 0.0
        f = lambda a: self._tls_exp2(a,chol_i,walker_up,walker_dn,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _d2_ut2_exp2_i(self,chol_i,walker_up,walker_dn,wave_data):
        x = 0.0
        f = lambda a: self._ut2_exp2(a,chol_i,walker_up,walker_dn,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _d2_tls_exp2(self,walker_up,walker_dn,ham_data,wave_data):
        norb = self.norb
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)
        d2_exp2_batch = jax.vmap(self._d2_tls_exp2_i, in_axes=(0,None,None,None))
        d2_exp2s = d2_exp2_batch(chol,walker_up,walker_dn,wave_data)
        h2 = jnp.sum(d2_exp2s)/2
        return h2

    @partial(jit, static_argnums=0)
    def _d2_ut2_exp2(self,walker_up,walker_dn,ham_data,wave_data):
        norb = self.norb
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)
        d2_exp2_batch = jax.vmap(self._d2_ut2_exp2_i, in_axes=(0,None,None,None))
        d2_exp2s = d2_exp2_batch(chol,walker_up,walker_dn,wave_data)
        h2 = jnp.sum(d2_exp2s)/2
        return h2

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        t1 = <exp(T1)HF|walker>/<HF|walker>
        t2 = <exp(T1)HF|T1+T2|walker>/<HF|walker>
        e0 = <exp(T1)HF|h1+h2|walker>/<HF|walker>
        e1 = <exp(T1)HF|(T1+T2)(h1+h2)|walker>/<HF|walker>
        '''

        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)

        # e0 = <exp(T1)HF|h1+h2|walker>/<HF|walker> #
        # one body
        x = 0.0
        f1 = lambda a: self._tls_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        t1, d_exp1_0 = jvp(f1, [x], [1.0])

        # two body
        d2_exp2_0 = self._d2_tls_exp2(walker_up,walker_dn,ham_data,wave_data)

        e0 = d_exp1_0 + d2_exp2_0
        
        # e1 = <exp(T1)HF|(T1+T2)(h1+h2)|walker>/<HF|walker>
        # one body
        x = 0.0
        f1 = lambda a: self._ut2_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        t2, d_exp1_1 = jvp(f1, [x], [1.0])

        # two body
        d2_exp2_1 = self._d2_ut2_exp2(walker_up,walker_dn,ham_data,wave_data)

        e1 = d_exp1_1 + d2_exp2_1

        # o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        return jnp.real(t1), jnp.real(t2), jnp.real(e0), jnp.real(e1)

    @singledispatchmethod
    def calc_energy_pt(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        raise NotImplementedError("Walker type not supported")

    @calc_energy_pt.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t1, t2, e0, e1
    
    @calc_energy_pt.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers, walkers, ham_data, wave_data)
        return t1, t2, e0, e1

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))