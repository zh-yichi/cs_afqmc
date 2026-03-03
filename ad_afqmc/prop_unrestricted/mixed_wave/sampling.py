import jax
import jax.numpy as jnp
from jax import jit, lax, random
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from ad_afqmc.prop_unrestricted.sampling import sampler
from ad_afqmc.prop_unrestricted.propagation import propagator

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
        og = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = og
        otg, eg, et = trial.calc_energy_mixed(prop_data["walkers"],ham_data,wave_data)
        de, do = trial.calc_stoccsd_cr(prop_data["walkers"], ham_data, wave_data)

        ot = otg * og
        ot_cr = ot + do
        et_cr = (et * ot + de) / ot_cr

        otg = jnp.real(otg)
        eg = jnp.real(eg)
        et = jnp.real(et)
        ot_cr = jnp.real(ot_cr)
        et_cr = jnp.real(et_cr)

        eg = jnp.where(
            jnp.abs(eg - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            eg,
        )
        
        wt = prop_data["weights"]
        wp = wt * otg
        wp_cr = wt * ot_cr / og

        blk_wt = jnp.sum(wt)
        blk_wp = jnp.sum(wp)
        blk_wp_cr = jnp.sum(wp_cr)
        blk_eg = jnp.sum(eg * wt) / blk_wt
        blk_et = jnp.sum(et * wp) / blk_wp
        blk_et_cr = jnp.sum(et_cr * wp_cr) / blk_wp_cr

        prop_data["pop_control_ene_shift"] = (
                0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_eg
            )

        return prop_data, (blk_wt, blk_wp, blk_wp_cr, blk_eg, blk_et, blk_et_cr)

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
        
        prop_data, (blk_wt, blk_wp, blk_wp_cr, blk_eg, blk_et, blk_et_cr) \
            = lax.scan(
            _block_scan_wrapper, prop_data, None, length = self.n_ene_blocks
        )
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt, blk_wp, blk_wp_cr, blk_eg, blk_et, blk_et_cr)

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
        prop_data, (blk_wt, blk_wp, blk_wp_cr, blk_eg, blk_et, blk_et_cr)\
            = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length = self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        wp = jnp.sum(blk_wp)
        wp_cr = jnp.sum(blk_wp_cr)
        eg = jnp.sum(blk_eg * blk_wt) / wt
        et = jnp.sum(blk_et * blk_wp) / wp
        et_cr = jnp.sum(blk_et_cr * blk_wp_cr) / wp_cr

        return prop_data, (wt, wp, wp_cr, eg, et, et_cr)
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_stoccsd2(sampler):
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

        # raondom fields_x for T2 decomposition
        xtaus = trial.get_xtaus(prop_data, wave_data, prop)

        prop_data = prop.orthonormalize_walkers(prop_data)
        overlap_hf = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = overlap_hf
        overlap_ci, energy_ci = trial.calc_energy_ci(prop_data["walkers"], ham_data, wave_data)
        numerator_cr = trial.calc_numerator(prop_data["walkers"], xtaus, ham_data, wave_data)
        denominator_cr = trial.calc_denominator(prop_data["walkers"], xtaus, wave_data)

        num_ci = overlap_ci * energy_ci / overlap_hf
        den_ci = overlap_ci / overlap_hf
        num_cr = numerator_cr / overlap_hf
        den_cr = denominator_cr / overlap_hf

        whf = prop_data["weights"]

        blk_whf = jnp.sum(whf)
        blk_num_ci = jnp.sum(whf * num_ci) / blk_whf
        blk_den_ci = jnp.sum(whf * den_ci) / blk_whf
        blk_num_cr = jnp.sum(whf * num_cr) / blk_whf
        blk_den_cr = jnp.sum(whf * den_cr) / blk_whf

        return prop_data, (blk_whf, blk_num_ci, blk_den_ci, blk_num_cr, blk_den_cr)

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
        
        prop_data, (blk_whf, blk_num_ci, blk_den_ci, blk_num_cr, blk_den_cr) \
            = lax.scan(
            _block_scan_wrapper, prop_data, None, length = self.n_ene_blocks
        )
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_whf, blk_num_ci, blk_den_ci, blk_num_cr, blk_den_cr)
    
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
        prop_data, (blk_whf, blk_num_ci, blk_den_ci, blk_num_cr, blk_den_cr) \
            = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length = self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )
        
        whf = jnp.sum(blk_whf)
        num_ci = jnp.sum(blk_whf * blk_num_ci) / whf
        den_ci = jnp.sum(blk_whf * blk_den_ci) / whf
        num_cr = jnp.sum(blk_whf * blk_num_cr) / whf
        den_cr = jnp.sum(blk_whf * blk_den_cr) / whf

        return prop_data, (whf, num_ci, den_ci, num_cr, den_cr)
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    
@dataclass
class sampler_ustoccsd2(sampler_stoccsd2):
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

        xtaus = trial.get_xtaus(prop_data, wave_data, prop)

        prop_data = prop.orthonormalize_walkers(prop_data)
        overlap_hf = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = overlap_hf
        overlap_ci, energy_ci = trial.calc_energy_cid(prop_data["walkers"], ham_data, wave_data)
        numerator_cr, denominator_cr = trial.calc_correction(prop_data["walkers"], xtaus, ham_data, wave_data)

        num_ci = overlap_ci * energy_ci / overlap_hf
        den_ci = overlap_ci / overlap_hf
        num_cr = numerator_cr / overlap_hf
        den_cr = denominator_cr / overlap_hf

        whf = prop_data["weights"]

        blk_whf = jnp.sum(whf)
        blk_num_ci = jnp.sum(whf * num_ci) / blk_whf
        blk_den_ci = jnp.sum(whf * den_ci) / blk_whf
        blk_num_cr = jnp.sum(whf * num_cr) / blk_whf
        blk_den_cr = jnp.sum(whf * den_cr) / blk_whf

        return prop_data, (blk_whf, blk_num_ci, blk_den_ci, blk_num_cr, blk_den_cr)
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))