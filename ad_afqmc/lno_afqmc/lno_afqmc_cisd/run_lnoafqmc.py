from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, stat_utils
from ad_afqmc.lno_afqmc import sampling, lno_afqmc
import time
import argparse

from ad_afqmc import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print = partial(print, flush=True)

ham_data, ham, prop, trial, wave_data, sampler, options, _ = (
    lno_afqmc._prep_afqmc())

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### initialize propagation
seed = options["seed"]
init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

eo = trial.calc_orb_energy(prop_data['walkers'], ham_data, wave_data)
eo_init = jnp.array(jnp.sum(eo)/ prop.n_walkers)

comm.Barrier()
if rank == 0:
    e_init = prop_data["e_estimate"]
    print('# \n')
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print("# Equilibration sweeps:")
    print("#   Iter \t Energy \t eorb \t \t time")
    print(f"  {0:5d} \t {e_init:.6f} \t {eo_init:.6f} \t"
          f"  {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler(
    n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10, n_chol = sampler.n_chol)

for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_e, blk_eo) = \
        sampler_eq.propagate_phaseless(ham_data, prop, prop_data, trial, wave_data)

    blk_wt = np.array([blk_wt], dtype="float64")
    blk_e = np.array([blk_e], dtype="float64")
    blk_eo = np.array([blk_eo], dtype="float64")

    gather_wt = None
    gather_e = None
    gather_eo = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_e = np.zeros(size, dtype="float64")
        gather_eo = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_e, gather_e, root=0)
    comm.Gather(blk_eo, gather_eo, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None
        blk_wt= np.sum(gather_wt)
        blk_e = np.sum(gather_wt * gather_e) / blk_wt
        blk_eo = np.sum(gather_wt * gather_eo) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_e, root=0)
    comm.Bcast(blk_eo, root=0)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (0.9 * prop_data["e_estimate"] + 0.1 * blk_e)

    comm.Barrier()
    if rank == 0:
        print(f"  {n:5d} \t {blk_e:.6f} \t {blk_eo:.6f} \t "
              f"  {time.time() - init_time:.2f} "
        )
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter  Energy \t error \t \t e_orb \t \t error \t \t time")
comm.Barrier()

glb_blk_wt = None
glb_blk_e = None
glb_blk_eo = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_e = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_eo = np.zeros(size * sampler.n_blocks,dtype="float64")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_e, blk_eo) = \
        sampler.propagate_phaseless(ham_data, prop, prop_data, trial, wave_data)
    
    blk_wt = np.array([blk_wt], dtype="float64")
    blk_e = np.array([blk_e], dtype="float64")
    blk_eo = np.array([blk_eo], dtype="float64")

    gather_wt = None
    gather_e = None
    gather_eo = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_e = np.zeros(size, dtype="float64")
        gather_eo = np.zeros(size, dtype="float64") 
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_e, gather_e, root=0)
    comm.Gather(blk_eo, gather_eo, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_e[n * size : (n + 1) * size] = gather_e
        glb_blk_eo[n * size : (n + 1) * size] = gather_eo

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_e = np.sum(gather_wt * gather_e) / blk_wt
        blk_eo = np.sum(gather_wt * gather_eo) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_e, root=0)
    comm.Bcast(blk_eo, root=0)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_e

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:
            e, e_err = \
                stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_e[: (n + 1) * size],
                    neql=0,
                )
            eo, eo_err = \
                stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_eo[: (n + 1) * size],
                    neql=0,
                )
            
            if e_err is not None:
                e_err = f"{e_err:.6f}"
            else:
                e_err = f"  {e_err}  "

            if eo_err is not None:
                eo_err = f"{eo_err:.6f}"
            else:
                eo_err = f"  {eo_err}  "
            
            e = f"{e:.6f}"
            eo = f"{eo:.6f}"
            
            print(f"  {n:4d} \t {e} \t {e_err} \t {eo} \t {eo_err} \t"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()

comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None

    samples_clean, idx = stat_utils.reject_outliers(
        np.stack((
                glb_blk_wt,
                glb_blk_e,
                glb_blk_eo,
                )).T,
                1,
            )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_e = samples_clean[:, 1]
    glb_blk_eo = samples_clean[:, 2]

    e, e_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_e,neql=0,printQ=True)
    eo, eo_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_eo,neql=0,printQ=True)

    if e_err is not None:
        e_err = f"{e_err:.6f}"
    else:
        e_err = f"  {e_err}  "

    if eo_err is not None:
        eo_err = f"{eo_err:.6f}"
    else:
        eo_err = f"  {eo_err}  "
    
    e = f"{e:.6f}"
    eo = f"{eo:.6f}"
    
    print(f"# Final Results")
    print(f"# AFQMC/CISD energy: {e} +/- {e_err}")
    print(f"# AFQMC/CISD E_Orbital: {eo} +/- {eo_err}")
    print(f"# total run time: {time.time() - init_time:.2f}")

comm.Barrier()
