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
        eorb, eorb0, eorb12, oorb12 = trial.calc_orb_energy(prop_data["walkers"], ham_data, wave_data)

        blk_wt = jnp.sum(prop_data["weights"])
        blk_e = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
        blk_eo = jnp.sum(eorb * prop_data["weights"]) / blk_wt
        blk_eo0 = jnp.sum(eorb0 * prop_data["weights"]) / blk_wt
        blk_eo12 = jnp.sum(eorb12 * prop_data["weights"]) / blk_wt
        blk_oo12 = jnp.sum(oorb12 * prop_data["weights"]) / blk_wt

        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_e
        )
        
        return prop_data, (blk_wt, blk_e, blk_eo, blk_eo0, blk_eo12, blk_oo12)
    

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

        prop_data, (blk_wt, blk_e, blk_eo, blk_eo0, blk_eo12, blk_oo12) = \
            lax.scan(_block_scan_wrapper, prop_data, None, length=self.n_ene_blocks)

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt, blk_e, blk_eo, blk_eo0, blk_eo12, blk_oo12)
    
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

        prop_data, (blk_wt, blk_e, blk_eo, blk_eo0, blk_eo12, blk_oo12) = \
            lax.scan(_sr_block_scan_wrapper, prop_data, None, length=self.n_sr_blocks)
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        e = jnp.sum(blk_e * blk_wt) / wt
        eo = jnp.sum(blk_eo * blk_wt) / wt
        eo0 = jnp.sum(blk_eo0 * blk_wt) / wt
        eo12 = jnp.sum(blk_eo12 * blk_wt) / wt
        oo12 = jnp.sum(blk_oo12 * blk_wt) / wt

        return prop_data, (wt, e, eo, eo0, eo12, oo12)

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

        eorb0, eorb012, torb12, ecorr \
            = trial.calc_orb_energy(prop_data["walkers"], ham_data, wave_data)
        
        ecorr = jnp.where(
            jnp.abs(ham_data['E0'] + ecorr
                    - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
                      prop_data["e_estimate"] - ham_data['E0'], ecorr)

        blk_wt = jnp.sum(prop_data["weights"])
        blk_ecorr = jnp.sum(ecorr * prop_data["weights"]) / blk_wt
        blk_eorb0 = jnp.sum(eorb0 * prop_data["weights"]) / blk_wt
        blk_eorb012 = jnp.sum(eorb012 * prop_data["weights"]) / blk_wt
        blk_torb12 = jnp.sum(torb12 * prop_data["weights"]) / blk_wt

        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * (blk_ecorr + ham_data['E0'])
        )
        
        return prop_data, (blk_wt, blk_ecorr, blk_eorb0, blk_eorb012, blk_torb12)
    

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

        prop_data, (blk_wt, blk_ecorr, blk_eorb0, blk_eorb012, blk_torb12) = \
            lax.scan(_block_scan_wrapper, prop_data, None, length=self.n_ene_blocks)

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt, blk_ecorr, blk_eorb0, blk_eorb012, blk_torb12)
    
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

        prop_data, (blk_wt, blk_ecorr, blk_eorb0, blk_eorb012, blk_torb12) = \
            lax.scan(_sr_block_scan_wrapper, prop_data, None, length=self.n_sr_blocks)
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        ecorr = jnp.sum(blk_ecorr * blk_wt) / wt
        eorb0 = jnp.sum(blk_eorb0 * blk_wt) / wt
        eorb012 = jnp.sum(blk_eorb012 * blk_wt) / wt
        torb12 = jnp.sum(blk_torb12 * blk_wt) / wt

        return prop_data, (wt, ecorr, eorb0, eorb012, torb12)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))