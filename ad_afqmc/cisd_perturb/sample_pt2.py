import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Tuple
import numpy as np
import pickle
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from ad_afqmc import wavefunctions
from ad_afqmc.hamiltonian import hamiltonian
from ad_afqmc.propagation import propagator
from ad_afqmc.wavefunctions import wave_function
from ad_afqmc.sampling import sampler
from ad_afqmc.cisd_perturb import ccsd_pt


@partial(jit, static_argnums=(0, 1))
def propagate(
    prop: propagator,
    trial: wave_function,
    ham_data: dict,
    prop_data: dict,
    fields: jax.Array,
    wave_data: dict,
) -> dict:
    """Phaseless AFQMC propagation. RHF guide
    Args:
        trial: trial wave function handler
        ham_data: dictionary containing the Hamiltonian data
        prop_data: dictionary containing the propagation data
        fields: auxiliary fields
        wave_data: wave function data

    Returns:
        prop_data: dictionary containing the updated propagation data
    """
    ### evaluate overlap & force bias with HF guide wave 
    guide = wavefunctions.rhf(trial.norb,trial.nelec,trial.n_batch)
    force_bias = guide.calc_force_bias(prop_data["walkers"], ham_data, wave_data)
    field_shifts = -jnp.sqrt(prop.dt) * (1.0j * force_bias - ham_data["mf_shifts"])
    shifted_fields = fields - field_shifts
    shift_term = jnp.sum(shifted_fields * ham_data["mf_shifts"], axis=1)
    fb_term = jnp.sum(
        fields * field_shifts - field_shifts * field_shifts / 2.0, axis=1
    )

    prop_data["walkers"] = prop._apply_trotprop(
        ham_data, prop_data["walkers"], shifted_fields
    )

    overlaps_new = guide.calc_overlap(prop_data["walkers"], wave_data)
    imp_fun = (
        jnp.exp(
            -jnp.sqrt(prop.dt) * shift_term
            + fb_term
            + prop.dt * (prop_data["pop_control_ene_shift"] + ham_data["h0_prop"])
        )
        * overlaps_new
        / prop_data["overlaps"]
    )
    theta = jnp.angle(
        jnp.exp(-jnp.sqrt(prop.dt) * shift_term)
        * overlaps_new
        / prop_data["overlaps"]
    )
    imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
    imp_fun_phaseless = jnp.array(
        jnp.where(jnp.isnan(imp_fun_phaseless), 0.0, imp_fun_phaseless)
    )
    imp_fun_phaseless = jnp.where(
        imp_fun_phaseless < 1.0e-3, 0.0, imp_fun_phaseless
    )
    imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100.0, 0.0, imp_fun_phaseless)

    prop_data["weights"] = imp_fun_phaseless * prop_data["weights"]
    prop_data["weights"] = jnp.array(
        jnp.where(prop_data["weights"] > 100, 0.0, prop_data["weights"])
    )
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
        jnp.log(jnp.sum(prop_data["weights"]) / prop.n_walkers) / prop.dt
    )
    prop_data["overlaps"] = overlaps_new
    return prop_data

@partial(jit, static_argnums=(3, 4))
def _step_scan(
    prop_data: dict,
    fields: jax.Array,
    ham_data: dict,
    prop: propagator,
    trial: wave_function,
    wave_data: dict,
) -> Tuple[dict, jax.Array]:
    """Phaseless propagation scan function over steps."""
    prop_data = propagate(prop, trial, ham_data, prop_data, fields, wave_data)
    return prop_data, fields

@partial(jit, static_argnums=(2,3,5))
def _block_scan(
    prop_data: dict,
    ham_data: dict,
    prop: propagator,
    trial: wave_function,
    wave_data: dict,
    sample: sampler,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    """Block scan function. Propagation and energy calculation."""
    prop_data["key"], subkey = random.split(prop_data["key"])
    fields = random.normal(
        subkey,
        shape=(
            sample.n_prop_steps,
            prop.n_walkers,
            ham_data["chol"].shape[0],
        ),
    )
    _step_scan_wrapper = lambda x, y: _step_scan(
        x, y, ham_data, prop, trial, wave_data
    )
    prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
    prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
        prop_data["weights"]
    )
    guide = wavefunctions.rhf(trial.norb,trial.nelec,trial.n_batch)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data["overlaps"] = guide.calc_overlap(prop_data["walkers"], wave_data)
    e0, e1, t = ccsd_pt.ccsd_walker_energy_pt(
        prop_data["walkers"],ham_data,wave_data,trial)

    wt = prop_data["weights"]
    blk_wt = jnp.sum(wt)
    blk_t = jnp.sum(t*wt)/blk_wt
    blk_e0 = jnp.sum(e0*wt)/blk_wt
    blk_e1 = jnp.sum(e1*wt)/blk_wt

    # h0 = ham_data['h0']
    # ept = jnp.real(h0 + e0 + e1 - blk_t*e0)
    # ept = jnp.where(
    #     jnp.abs(ept - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
    #     prop_data["e_estimate"],
    #     ept,
    # )
    # blk_ept = jnp.sum(ept*wt)/blk_wt

    # prop_data["pop_control_ene_shift"] = (
    #     0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_ept
    # )
    return prop_data, (blk_wt, blk_t, blk_e0, blk_e1)

@partial(jit, static_argnums=(2,3,5))
def _sr_block_scan(
    prop_data: dict,
    ham_data: dict,
    prop: propagator,
    trial: wave_function,
    wave_data: dict,
    sample: sampler,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    
    guide = wavefunctions.rhf(trial.norb,trial.nelec,trial.n_batch)
    
    def _block_scan_wrapper(x,_):
        return _block_scan(x,ham_data,prop,trial,wave_data,sample)
    
    prop_data, (blk_wt, blk_t, blk_e0, blk_e1)= lax.scan(
        _block_scan_wrapper, prop_data, None, length=sample.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = guide.calc_overlap(prop_data["walkers"], wave_data)
    return prop_data, (blk_wt, blk_t, blk_e0, blk_e1)

@partial(jit, static_argnums=(2, 3, 5))
def propagate_phaseless(
    prop_data: dict,
    ham_data: dict,
    prop: propagator,
    trial: wave_function,
    wave_data: dict,
    sample: sampler,
) -> Tuple[jax.Array, dict]:
    def _sr_block_scan_wrapper(x,_):
        return _sr_block_scan(x, ham_data, prop, trial, wave_data, sample)
    
    guide = wavefunctions.rhf(trial.norb,trial.nelec,trial.n_batch)

    prop_data["overlaps"] = guide.calc_overlap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data, (blk_wt, blk_t, blk_e0, blk_e1) = lax.scan(
        _sr_block_scan_wrapper, prop_data, None, length=sample.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sample.n_sr_blocks * sample.n_ene_blocks * prop.n_walkers
    )

    wt = jnp.sum(blk_wt)
    t = jnp.sum(blk_t * blk_wt) / wt
    e0 = jnp.sum(blk_e0 * blk_wt) / wt
    e1 = jnp.sum(blk_e1 * blk_wt) / wt

    return prop_data, (wt, t, e0, e1)

def run_afqmc_ccsd_pt(options,nproc=None,option_file='options.bin'):
    options["dt"] = options.get("dt", 0.005)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse", "2rdm"]
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", True)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", "cisd")
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["ene0"] = options.get("ene0",0)
    options["use_gpu"] = options.get("use_gpu", False)

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    use_gpu = options["use_gpu"]
    if use_gpu:
        print(f'# running AFQMC on GPU')
        gpu_flag = "--use_gpu"
        mpi_prefix = ""
        nproc = None
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)  
    script = f"{dir_path}/run_afqmc_ccsd_pt.py"
    # script = f"{dir_path}/run_afqmc_cisd_pt.py"
    
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f"{mpi_prefix} python {script} {gpu_flag} |tee afqmc_ccsd_pt.out"
    )