import os
from functools import partial
from typing import Tuple
import pickle
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from ad_afqmc.propagation import propagator
from ad_afqmc.wavefunctions import wave_function
from ad_afqmc.sampling import sampler

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
    _step_scan_wrapper = lambda x, y: sample._step_scan(
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
    
    # e0 = jnp.where(
    #     jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
    #     prop_data["e_estimate"],
    #     e0,
    # )
    
    wt = prop_data["weights"]

    blk_wt = jnp.sum(wt)
    blk_t1 = jnp.sum(t1*wt)/blk_wt
    blk_t2 = jnp.sum(t2*wt)/blk_wt
    blk_e0 = jnp.sum(e0*wt)/blk_wt
    blk_e1 = jnp.sum(e1*wt)/blk_wt

    # blk_ept = blk_e0 + blk_e1 + blk_t * (blk_e0 - ham_data['h0'])
    # prop_data["pop_control_ene_shift"] = (
    #         0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_ept
    #     )

    return prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1)

@partial(jit, static_argnums=(2,3,5))
def _sr_block_scan(
    prop_data: dict,
    ham_data: dict,
    prop: propagator,
    trial: wave_function,
    wave_data: dict,
    sample: sampler,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
        
    def _block_scan_wrapper(x,_):
        return _block_scan(x,ham_data,prop,trial,wave_data,sample)
    
    prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1) = lax.scan(
        _block_scan_wrapper, prop_data, None, length=sample.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    return prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1)

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

    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1) = lax.scan(
        _sr_block_scan_wrapper, prop_data, None, length=sample.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sample.n_sr_blocks * sample.n_ene_blocks * prop.n_walkers
    )

    wt = jnp.sum(blk_wt)
    t1 = jnp.sum(blk_t1 * blk_wt) / wt
    t2 = jnp.sum(blk_t2 * blk_wt) / wt
    e0 = jnp.sum(blk_e0 * blk_wt) / wt
    e1 = jnp.sum(blk_e1 * blk_wt) / wt

    return prop_data, (wt, t1, t2, e0, e1)

def run_afqmc(options,nproc=None,
              option_file='options.bin',
              script='run_afqmc_ccsd_pt2.py'):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    use_gpu = options["use_gpu"]
    if use_gpu:
        print(f'# running AFQMC on GPU')
        gpu_flag = "--use_gpu"
        mpi_prefix = ""
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)  
    script = f"{dir_path}/{script}"
    print(f'# AFQMC script: {script}')
    
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f"{mpi_prefix} python {script} {gpu_flag} |tee afqmc_ccsd_pt.out"
    )