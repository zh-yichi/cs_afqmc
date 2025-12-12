from dataclasses import dataclass
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from ad_afqmc.hamiltonian import hamiltonian
from ad_afqmc.prop_unrestricted.propagation import propagator
from ad_afqmc.prop_unrestricted.wavefunctions import wave_function

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
        trial: wave_function,
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
        trial: wave_function,
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

        eorb = trial.calc_orb_energy(prop_data["walkers"], ham_data, wave_data)

        blk_wt = jnp.sum(prop_data["weights"])
        blk_e = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
        blk_eo = jnp.sum(eorb * prop_data["weights"]) / blk_wt

        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_e
        )
        
        return prop_data, (blk_wt, blk_e, blk_eo)
    

    @partial(jit, static_argnums=(0, 3, 4))
    def _sr_block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:

        def _block_scan_wrapper(x,_):
            return self._block_scan(x,ham_data,prop,trial,wave_data)

        prop_data, (blk_wt, blk_e, blk_eo) = \
            lax.scan(_block_scan_wrapper, prop_data, None, length=self.n_ene_blocks)

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt, blk_e, blk_eo)
    
    @partial(jit, static_argnums=(0, 2, 4))
    def propagate_phaseless(
        self,
        ham_data: dict,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_block_scan_wrapper(x,_):
            return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

        prop_data, (blk_wt, blk_e, blk_eo) = \
            lax.scan(_sr_block_scan_wrapper, prop_data, None, length=self.n_sr_blocks)
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        e = jnp.sum(blk_e * blk_wt) / wt
        eo = jnp.sum(blk_eo * blk_wt) / wt

        return prop_data, (wt, e, eo)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class sampler_pt:
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
        trial: wave_function,
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
        trial: wave_function,
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

        eorb, teorb, torb, e0 \
            = trial.calc_eorb_pt(prop_data["walkers"], ham_data, wave_data)
        
        e0 = jnp.where(
            jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt), 
            prop_data["e_estimate"], e0)

        blk_wt = jnp.sum(prop_data["weights"])
        blk_eorb = jnp.sum(eorb * prop_data["weights"]) / blk_wt
        blk_teorb = jnp.sum(teorb * prop_data["weights"]) / blk_wt
        blk_torb = jnp.sum(torb * prop_data["weights"]) / blk_wt
        blk_e0 = jnp.sum(e0 * prop_data["weights"]) / blk_wt

        prop_data["pop_control_ene_shift"] = \
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_e0
        
        return prop_data, (blk_wt, blk_eorb, blk_teorb, blk_torb, blk_e0)
    

    @partial(jit, static_argnums=(0, 3, 4))
    def _sr_block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:

        def _block_scan_wrapper(x,_):
            return self._block_scan(x,ham_data,prop,trial,wave_data)

        prop_data, (blk_wt, blk_eorb, blk_teorb, blk_torb, blk_e0) = \
            lax.scan(_block_scan_wrapper, prop_data, None, length=self.n_ene_blocks)

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt, blk_eorb, blk_teorb, blk_torb, blk_e0)
    
    @partial(jit, static_argnums=(0, 2, 4))
    def propagate_phaseless(
        self,
        ham_data: dict,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_block_scan_wrapper(x,_):
            return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

        prop_data, (blk_wt, blk_eorb, blk_teorb, blk_torb, blk_e0) = \
            lax.scan(_sr_block_scan_wrapper, prop_data, None, length=self.n_sr_blocks)
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        eorb = jnp.sum(blk_eorb * blk_wt) / wt
        teorb = jnp.sum(blk_teorb * blk_wt) / wt
        torb = jnp.sum(blk_torb * blk_wt) / wt
        e0 = jnp.sum(blk_e0 * blk_wt) / wt

        return prop_data, (wt, eorb, teorb, torb, e0)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))