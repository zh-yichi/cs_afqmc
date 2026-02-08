from dataclasses import dataclass
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from ad_afqmc.hamiltonian import hamiltonian
from ad_afqmc.prop_unrestricted.propagation import propagator
# from ad_afqmc.prop_unrestricted.wavefunctions import wave_function
from ad_afqmc import sr, linalg_utils

@dataclass
class sampler:
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data: dict,
        fields: jax.Array,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, jax.Array]:
        """Phaseless propagation scan function over steps."""
        prop_data = prop.propagate(trial, ham_data, prop_data, fields, wave_data)
        return prop_data, fields

    @partial(jit, static_argnums=(0, 3, 4))
    def _block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
        """Block scan function. Propagation and energy calculation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
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
        block_weight = jnp.sum(prop_data["weights"])
        block_energy = jnp.sum(energy_samples * prop_data["weights"]) / block_weight
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
        )
        return prop_data, (block_energy, block_weight)
    

    @partial(jit, static_argnums=(0, 3, 4))
    def _sr_block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:

        def _block_scan_wrapper(x,_):
            return self._block_scan(x,ham_data,prop,trial,wave_data)

        prop_data, (block_energy, block_weight) = lax.scan(
            _block_scan_wrapper, prop_data, None, length=self.n_ene_blocks
        )

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (block_energy, block_weight)

    @partial(jit, static_argnums=(0, 1, 3, 5))
    def propagate_phaseless(
        self,
        ham: hamiltonian,
        ham_data: dict,
        prop: propagator,
        prop_data: dict,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_block_scan_wrapper(x,_):
            return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (block_energy, block_weight) = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length=self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )
        # return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data
        weight = jnp.sum(block_weight)
        energy = jnp.sum(block_energy * block_weight) / weight
        return prop_data, (weight, energy)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class sampler_group:
    group_size: int = 2
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0,))
    def group_sr(self, prop_data: dict) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        group_weights = prop_data["group_weights"]
        group_walkers = prop_data["walkers"].reshape(-1, self.group_size)
        print('walkers shape ', group_walkers.shape)

        group_walkers, group_weights = sr.stochastic_reconfiguration(
            group_walkers, group_weights, zeta
        )

        prop_data["walkers"] = group_walkers.reshape(-1)
        prop_data["group_weights"] = group_weights
        
        return prop_data

    @partial(jit, static_argnums=(0, 1, 2))
    def propagate_group(
        self,
        prop,
        trial,
        ham_data: dict,
        prop_data: dict,
        fields: jax.Array,
        wave_data: dict,
    ) -> dict:
        
        olps = trial.calc_overlap(prop_data["walkers"], wave_data)
        fbs = trial.calc_force_bias(prop_data["walkers"], ham_data, wave_data)

        olps = olps.reshape(-1, self.group_size)
        fbs = fbs.reshape(-1, self.group_size, self.n_chol)
        print('fb shape ', fbs.shape)
        print('olps shape ', olps.shape)
        group_olps = jnp.sum(olps, axis=1) 
        group_fbs = jnp.einsum('Gw,Gwg->Gg', olps, fbs)
        group_fbs = jnp.einsum('Gg,G->Gg', group_fbs, 1/group_olps)
        print('group_fb shape ', group_fbs.shape)
        print('group_olps shape ', group_olps.shape)
        # group_fbs = jnp.broadcast_to(group_fbs[:, None], (group_fbs.shape[0], self.group_size)).reshape(-1)

        field_shifts = -jnp.sqrt(prop.dt) * (1.0j * group_fbs - ham_data["mf_shifts"])
        print('field_shifts shape ', field_shifts.shape)

        field_shifts = jnp.broadcast_to(field_shifts[:, None, :], 
                                        (field_shifts.shape[0], self.group_size, self.n_chol)
                                        ).reshape(-1,self.n_chol)
        
        print('field_shifts shape ', field_shifts.shape)

        shifted_fields = fields - field_shifts
        # shifted_fields = fields - jnp.broadcast_to(
        #     field_shifts[:, None, :], 
        #     (field_shifts.shape[0], self.group_size, self.n_chol)
        #     ).reshape(-1,self.n_chol)
        
        shift_term = jnp.sum(shifted_fields * ham_data["mf_shifts"], axis=1)
        fb_term = jnp.sum(
            fields * field_shifts - field_shifts * field_shifts / 2.0, axis=1
        )

        prop_data["walkers"] = prop._apply_trotprop(
            ham_data, prop_data["walkers"], shifted_fields
        )

        olps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        # olps_new = olps_new.reshape(-1, self.group_size)
        # group_olps_new = jnp.sum(olps_new, axis=1) 

        # I(x,xbar,walkers,trial) = <trial|walkers_new>/<trial|walkers_old> 
        #                                 * exp(x_g xbar_g - 1/2 xbar_g xbar_g)
        print('shift_term shape ', shift_term.shape)
        print('fb_term shape ', fb_term.shape)
        # print('gp_olp_new shape ', group_olps_new.shape)
        # print('gp_olp shape ', group_olps.shape)
        imp_fun = (
            jnp.exp(
                -jnp.sqrt(prop.dt) * shift_term
                + fb_term # pop_control_ene_shift = estimated ground state energy
                + prop.dt * (prop_data["pop_control_ene_shift"] + ham_data["h0_prop"])
                ) * olps_new
            )
        imp_fun = jnp.sum(imp_fun.reshape(-1, self.group_size), axis=1) / group_olps
        
        theta = jnp.exp(-jnp.sqrt(prop.dt) * shift_term) * olps_new
        theta = jnp.angle(jnp.sum(theta.reshape(-1, self.group_size), axis=1) / group_olps)
        print('theta ', theta)

        imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
        imp_fun_phaseless = jnp.array(
            jnp.where(jnp.isnan(imp_fun_phaseless), 0.0, imp_fun_phaseless)
        )
        imp_fun_phaseless = jnp.where(
            imp_fun_phaseless < 1.0e-3, 0.0, imp_fun_phaseless
        )
        imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100.0, 0.0, imp_fun_phaseless)
        print('imp_fun_phaseless', imp_fun_phaseless)

        # prop_data["imp_fun"] = imp_fun_phaseless
        prop_data["group_weights"] = imp_fun_phaseless * prop_data["group_weights"]
        prop_data["group_weights"] = jnp.array(
            jnp.where(prop_data["group_weights"] > 100, 0.0, prop_data["group_weights"])
        )
        print('group weights ', prop_data["group_weights"])

        prop_data["pop_control_ene_shift"] \
            = prop_data["e_estimate"] - 0.1 * jnp.array(
                jnp.log(jnp.sum(prop_data["group_weights"]) 
                    / (prop.n_walkers / self.group_size)) 
                    / prop.dt
                    )
        prop_data["overlaps"] = olps_new

        return prop_data

    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data: dict,
        fields: jax.Array,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, jax.Array]:
        """Phaseless propagation scan function over steps."""
        prop_data = self.propagate_group(prop, trial, ham_data, prop_data, fields, wave_data)
        return prop_data, fields

    @partial(jit, static_argnums=(0, 3, 4))
    def _block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
        """Block scan function. Propagation and energy calculation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_groups"] += prop_data["group_weights"].size - jnp.count_nonzero(
            prop_data["group_weights"]
        )
        prop_data = prop.orthonormalize_walkers(prop_data)

        # prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        olps = trial.calc_overlap(prop_data["walkers"], wave_data)
        ens = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        print('walkers shape ', prop_data["walkers"].shape)

        olps = olps.reshape(-1, self.group_size)
        ens = ens.reshape(-1, self.group_size)
        group_olps = jnp.sum(olps, axis=1) 
        group_ens = jnp.real(jnp.sum(olps * ens, axis=1) / group_olps)

        group_ens = jnp.where(
            jnp.abs(group_ens - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            group_ens,
        )

        wt = jnp.sum(prop_data["group_weights"])
        en = jnp.sum(group_ens * prop_data["group_weights"]) / wt
        
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * en
        )

        return prop_data, (en, wt)
    

    @partial(jit, static_argnums=(0, 3, 4))
    def _sr_block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:

        def _block_scan_wrapper(x,_):
            return self._block_scan(x,ham_data,prop,trial,wave_data)

        prop_data, (block_energy, block_weight) = lax.scan(
            _block_scan_wrapper, prop_data, None, length=self.n_ene_blocks
        )
        print('walkers shape ', prop_data["walkers"].shape)
        # prop_data = self.group_sr(prop_data)
        print('walkers shape ', prop_data["walkers"].shape)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (block_energy, block_weight)

    @partial(jit, static_argnums=(0, 1, 3, 5))
    def propagate_phaseless(
        self,
        ham: hamiltonian,
        ham_data: dict,
        prop: propagator,
        prop_data: dict,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_block_scan_wrapper(x,_):
            return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_groups"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (block_energy, block_weight) = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length=self.n_sr_blocks
        )
        prop_data["n_killed_groups"] /= (
            self.n_sr_blocks * self.n_ene_blocks * (prop.n_walkers // self.group_size)
        )

        weight = jnp.sum(block_weight)
        energy = jnp.sum(block_energy * block_weight) / weight
        return prop_data, (weight, energy)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))



@dataclass
class sampler_abs:
    n_prop_steps: int = 10
    n_sr_blocks: int = 10 
    n_ene_blocks: int = 10
    n_blocks: int = 50
    n_chol: int = 0
    
    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data: dict,
        fields: jax.Array,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, jax.Array]:
        """Phaseless propagation scan function over steps."""
        prop_data = prop.propagate_free(trial, ham_data, prop_data, fields)
        prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        norms = norms[0] * norms[1]
        norms_abs = jnp.abs(norms)
        phase = norms / norms_abs
        # weights = jnp.real(norms[0] * norms[1])
        nwalker = int(prop_data["weights"].shape[0])
        prop_data["weights"] *= norms_abs
        prop_data["weights"] = nwalker * prop_data["weights"] / jnp.sum(prop_data["weights"])
        nocca, noccb = prop_data["walkers"][0].shape[-1], prop_data["walkers"][1].shape[-1]
        phase_mo = jnp.stack((phase**(1/(2.0*nocca)), phase**(1/(2.0*noccb)))) # multiply the phase into the mo_coeff
        prop_data["walkers"] = prop._multiply_constant(prop_data["walkers"], phase_mo)

        # sr
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        return prop_data, fields
    
    @partial(jit, static_argnums=(0, 3, 4))
    def _sr_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,):
        """free propagation for a block of (n_prop_steps, n_walkers)."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
            )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)

        # prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        # norms = norms[0] * norms[1]
        # norms_abs = jnp.abs(norms)
        # phase = norms / norms_abs

        # nwalker = prop.n_walkers
        # prop_data["weights"] *= norms_abs # keep weight real
        # prop_data["weights"] = nwalker * prop_data["weights"] / jnp.sum(prop_data["weights"])

        # nocca, noccb = trial.nelec
        # phase_mo = jnp.stack((phase**(1/(2.0*nocca)), phase**(1/(2.0*noccb)))) # multiply the phase into coeff
        # prop_data["walkers"] = prop._multiply_constant(prop_data["walkers"], phase_mo)

        # # sr
        # prop_data = prop.stochastic_reconfiguration_local(prop_data)

        return prop_data, None

    @partial(jit, static_argnums=(0, 3, 4))
    def _ene_block(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_scan_wrapper(x,_):
            return self._sr_scan(x, ham_data, prop, trial, wave_data)

        prop_data, _ = lax.scan(
            _sr_scan_wrapper, prop_data, None, length=self.n_sr_blocks
        )

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        energy_samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )

        block_energy = jnp.real(
            jnp.sum(
                energy_samples * jnp.abs(prop_data["overlaps"]) * prop_data["weights"]
                ) / jnp.sum(jnp.abs(prop_data["overlaps"]) * prop_data["weights"])
            )
        
        block_weight = jnp.sum(jnp.abs(prop_data["overlaps"]) * prop_data["weights"])

        # prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy

        return prop_data, (block_weight, block_energy)

    
    @partial(jit, static_argnums=(0, 3, 4))
    def propagate_abs(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _ene_block_wrapper(x,_):
            return self._ene_block(x, ham_data, prop, trial, wave_data)

        prop_data, (block_weight, block_energy) = lax.scan(
            _ene_block_wrapper, prop_data, None, length=self.n_ene_blocks
        )

        weight = jnp.sum(block_weight)
        energy = jnp.sum(block_energy * block_weight) / weight

        return prop_data, (weight, energy)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class sampler_fp:
    n_prop_steps: int = 50
    n_eql_blocks: int = 10
    n_trj: int = 100
    n_chol: int = 0
    
    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data: dict,
        fields: jax.Array,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, jax.Array]:
        """Phaseless propagation scan function over steps."""
        prop_data = prop.propagate_free(trial, ham_data, prop_data, fields)
        prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        norms = norms[0] * norms[1]
        norms_abs = jnp.abs(norms)
        phase = norms / norms_abs
        # weights = jnp.real(norms[0] * norms[1])
        nwalker = int(prop_data["weights"].shape[0])
        prop_data["weights"] *= norms_abs
        prop_data["weights"] = nwalker * prop_data["weights"] / jnp.sum(prop_data["weights"])
        nocca, noccb = prop_data["walkers"][0].shape[-1], prop_data["walkers"][1].shape[-1]
        phase_mo = jnp.stack((phase**(1/(2.0*nocca)), phase**(1/(2.0*noccb)))) # multiply the phase into the mo_coeff
        prop_data["walkers"] = prop._multiply_constant(prop_data["walkers"], phase_mo)

        # sr
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        return prop_data, fields
    
    @partial(jit, static_argnums=(0, 3, 4))
    def fp_block(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict):
        """free propagation for a block of (n_prop_steps, n_walkers)."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
            )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        energy_samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)

        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )

        # blk_e = jnp.real(
        #     jnp.sum(
        #         energy_samples * prop_data["overlaps"] * prop_data["weights"]
        #         ) / jnp.sum(prop_data["overlaps"] * prop_data["weights"])
        #     )
        
        blk_e = jnp.sum(
                energy_samples * prop_data["overlaps"] * prop_data["weights"]
                ) / jnp.sum(prop_data["overlaps"] * prop_data["weights"])

        blk_w = jnp.sum(prop_data["overlaps"] * prop_data["weights"])

        return prop_data, (blk_w, blk_e)
    
    @partial(jit, static_argnums=(0, 3, 4))
    def scan_eql_blocks(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _block_scan_wrapper(x,_):
            return self.fp_block(x, ham_data, prop, trial, wave_data)

        prop_data, (blk_w, blk_e) = lax.scan(
            _block_scan_wrapper, prop_data, None, length=self.n_eql_blocks
        )

        return prop_data, (blk_w, blk_e)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_fpabs:
    n_prop_steps: int = 50
    n_eql_blocks: int = 10
    n_trj: int = 100
    n_chol: int = 0
    
    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data: dict,
        fields: jax.Array,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, jax.Array]:
        """Phaseless propagation scan function over steps."""
        prop_data = prop.propagate_free(trial, ham_data, prop_data, fields)
        prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        norms = norms[0] * norms[1]
        norms_abs = jnp.abs(norms)
        phase = norms / norms_abs
        # weights = jnp.real(norms[0] * norms[1])
        nwalker = int(prop_data["weights"].shape[0])
        prop_data["weights"] *= norms_abs
        prop_data["weights"] = nwalker * prop_data["weights"] / jnp.sum(prop_data["weights"])
        nocca, noccb = prop_data["walkers"][0].shape[-1], prop_data["walkers"][1].shape[-1]
        phase_mo = jnp.stack((phase**(1/(2.0*nocca)), phase**(1/(2.0*noccb)))) # multiply the phase into the mo_coeff
        prop_data["walkers"] = prop._multiply_constant(prop_data["walkers"], phase_mo)

        # sr
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        return prop_data, fields
    
    @partial(jit, static_argnums=(0, 3, 4))
    def fp_block(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict):
        """free propagation for a block of (n_prop_steps, n_walkers)."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
            )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        energy_samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)

        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )

        # blk_e = jnp.real(
        #     jnp.sum(
        #         energy_samples * prop_data["overlaps"] * prop_data["weights"]
        #         ) / jnp.sum(prop_data["overlaps"] * prop_data["weights"])
        #     )
        
        blk_e = jnp.real(
                    jnp.sum(
                        energy_samples * jnp.abs(prop_data["overlaps"]) * prop_data["weights"]
                        ) / jnp.sum(jnp.abs(prop_data["overlaps"]) * prop_data["weights"]
                            )
                        )

        blk_w = jnp.real(jnp.sum(jnp.abs(prop_data["overlaps"]) * prop_data["weights"]))

        return prop_data, (blk_w, blk_e)
    
    @partial(jit, static_argnums=(0, 3, 4))
    def scan_eql_blocks(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _block_scan_wrapper(x,_):
            return self.fp_block(x, ham_data, prop, trial, wave_data)

        prop_data, (blk_w, blk_e) = lax.scan(
            _block_scan_wrapper, prop_data, None, length=self.n_eql_blocks
        )

        return prop_data, (blk_w, blk_e)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_mixed(sampler):
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0,3,4))
    def _block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
        """Block scan function. Propagation and energy calculation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )

        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        otg, eg, et = trial.calc_energy_mixed(prop_data["walkers"],ham_data,wave_data)
        
        eg = jnp.where(
            jnp.abs(eg - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            eg,
        )
        
        wt = prop_data["weights"]
        wp = wt * otg

        blk_wt = jnp.sum(wt)
        blk_wp = jnp.sum(wp)
        blk_eg = jnp.sum(eg*wt) / blk_wt
        blk_et = jnp.sum(et*wp) / blk_wp

        prop_data["pop_control_ene_shift"] = (
                0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_eg
            )

        return prop_data, (blk_wt, blk_wp, blk_eg, blk_et)

    @partial(jit, static_argnums=(0,3,4))
    def _sr_block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
            
        def _block_scan_wrapper(x,_):
            return self._block_scan(x,ham_data,prop,trial,wave_data)
        
        prop_data, (blk_wt, blk_wp, blk_eg, blk_et)= lax.scan(
            _block_scan_wrapper, prop_data, None, length = self.n_ene_blocks
        )
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt, blk_wp, blk_eg, blk_et)

    @partial(jit, static_argnums=(0,3,4))
    def propagate_phaseless(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_block_scan_wrapper(x,_):
            return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (blk_wt, blk_wp, blk_eg, blk_et) = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length = self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        wp = jnp.sum(blk_wp)
        eg = jnp.sum(blk_eg * blk_wt) / wt
        et = jnp.sum(blk_et * blk_wp) / wp

        return prop_data, (wt, wp, eg, et)
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_pt(sampler):
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0,3,4))
    def _block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
        """Block scan function. Propagation and energy calculation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )

        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        t, e0, e1 = trial.calc_energy_pt(
            prop_data["walkers"],ham_data,wave_data)
        
        # e0 = jnp.where(
        #     jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
        #     prop_data["e_estimate"],
        #     e0,
        # )
        
        wt = prop_data["weights"]

        blk_wt = jnp.sum(wt)
        blk_t = jnp.sum(t*wt)/blk_wt
        blk_e0 = jnp.sum(e0*wt)/blk_wt
        blk_e1 = jnp.sum(e1*wt)/blk_wt

        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

        # blk_ept = blk_e0 + blk_e1 + blk_t * (blk_e0 - ham_data['h0'])
        # prop_data["pop_control_ene_shift"] = (
        #         0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_ept
        #     )

        return prop_data, (blk_wt, blk_t, blk_e0, blk_e1)

    @partial(jit, static_argnums=(0,3,4))
    def _sr_block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
            
        def _block_scan_wrapper(x,_):
            return self._block_scan(x,ham_data,prop,trial,wave_data)
        
        prop_data, (blk_wt, blk_t, blk_e0, blk_e1)= lax.scan(
            _block_scan_wrapper, prop_data, None, length = self.n_ene_blocks
        )
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt, blk_t, blk_e0, blk_e1)

    @partial(jit, static_argnums=(0,3,4))
    def propagate_phaseless(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_block_scan_wrapper(x,_):
            return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (blk_wt, blk_t, blk_e0, blk_e1) = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length = self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        t = jnp.sum(blk_t * blk_wt) / wt
        e0 = jnp.sum(blk_e0 * blk_wt) / wt
        e1 = jnp.sum(blk_e1 * blk_wt) / wt

        return prop_data, (wt, t, e0, e1)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))

@dataclass
class sampler_pt2(sampler):
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0,3,4))
    def _block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
        """Block scan function. Propagation and energy calculation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )

        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        t1, t2, e0, e1 = trial.calc_energy_pt(
            prop_data["walkers"],ham_data,wave_data)
        # ehf = jnp.real(trial.calc_energy(
        #      prop_data["walkers"],ham_data,wave_data))
        
        wt = prop_data["weights"]

        blk_wt = jnp.sum(wt)
        blk_t1 = jnp.sum(t1*wt)/blk_wt
        blk_t2 = jnp.sum(t2*wt)/blk_wt
        blk_e0 = jnp.sum(e0*wt)/blk_wt
        blk_e1 = jnp.sum(e1*wt)/blk_wt
        # blk_ehf = jnp.sum(e_hf*wt)/blk_wt

        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

        # blk_ept = (ham_data['h0'] + 1/blk_t1 * blk_e0 
        #            + 1/blk_t1 * blk_e1 - 1/blk_t1**2 * blk_t2 * blk_e0)
        # prop_data["pop_control_ene_shift"] = (
        #         0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_ept
        #     )

        return prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1)

    @partial(jit, static_argnums=(0,3,4))
    def _sr_block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
            
        def _block_scan_wrapper(x,_):
            return self._block_scan(x,ham_data,prop,trial,wave_data)
        
        prop_data, (blk_wt,blk_t1,blk_t2,blk_e0,blk_e1) = lax.scan(
            _block_scan_wrapper, prop_data, None, length = self.n_ene_blocks
        )
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt,blk_t1,blk_t2,blk_e0,blk_e1)

    @partial(jit, static_argnums=(0,3,4))
    def propagate_phaseless(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_block_scan_wrapper(x,_):
            return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (blk_wt,blk_t1,blk_t2,blk_e0,blk_e1) = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length = self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        t1 = jnp.sum(blk_t1 * blk_wt) / wt
        t2 = jnp.sum(blk_t2 * blk_wt) / wt
        e0 = jnp.sum(blk_e0 * blk_wt) / wt
        e1 = jnp.sum(blk_e1 * blk_wt) / wt
        # ehf = jnp.sum(blk_ehf * blk_wt) / wt

        return prop_data, (wt, t1, t2, e0, e1)


    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
