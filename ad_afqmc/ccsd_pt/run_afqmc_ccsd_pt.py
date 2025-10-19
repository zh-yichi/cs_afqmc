from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, sampling, stat_utils, mpi_jax
from ad_afqmc.ccsd_pt import sample_ccsd_pt
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

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
    mpi_jax._prep_afqmc())

# if options["use_gpu"]:
#     config.afqmc_config["use_gpu"] = True

# config.setup_jax()
# MPI = config.setup_comm()

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes


### initialize propagation
seed = options["seed"]
init_walkers = None
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
h0 = ham_data['h0']

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

comm.Barrier()
if rank == 0:
    t_init, e0_init, e1_init = trial._calc_energy_pt_restricted(
        prop_data['walkers'][0], ham_data, wave_data)
    ept_init = e0_init + e1_init - t_init*(e0_init-h0)
    print('# \n')
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print("# Equilibration sweeps:")
    print("#   Iter \t energy \t Walltime")
    print(f"  {0:5d} \t {ept_init:.6f} \t {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler(n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10)
for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_t, blk_e0, blk_e1) =\
        sample_ccsd_pt.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler_eq)

    blk_wt = np.array([blk_wt], dtype="float32") 
    blk_t = np.array([blk_t], dtype="float32")
    blk_e0 = np.array([blk_e0], dtype="float32")
    blk_e1 = np.array([blk_e1], dtype="float32")

    blk_wt_t = np.array([blk_t * blk_wt], dtype="float32")
    blk_wt_e0 = np.array([blk_e0 * blk_wt], dtype="float32")
    blk_wt_e1 = np.array([blk_e1 * blk_wt], dtype="float32")

    tot_blk_wt = np.zeros(1, dtype="float32")
    tot_blk_t = np.zeros(1, dtype="float32")
    tot_blk_e0 = np.zeros(1, dtype="float32")
    tot_blk_e1 = np.zeros(1, dtype="float32")

    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_blk_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_t, MPI.FLOAT],
        [tot_blk_t, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_e0, MPI.FLOAT],
        [tot_blk_e0, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_e1, MPI.FLOAT],
        [tot_blk_e1, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )

    comm.Barrier()
    if rank == 0:
        blk_wt = tot_blk_wt
        blk_t = tot_blk_t / tot_blk_wt
        blk_e0 = tot_blk_e0 / tot_blk_wt
        blk_e1 = tot_blk_e1 / tot_blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_t, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)
    
    blk_ept = blk_e0 + blk_e1 - blk_t * (blk_e0-h0)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
         0.9 * prop_data["e_estimate"] + 0.1 * blk_ept[0]
         )
    # comm.Barrier()

    comm.Barrier()
    if rank == 0:
        print(
            f"  {n:5d} \t {blk_ept[0]:.6f} \t {time.time() - init_time:.2f} "
        )
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t energy \t error \t \t Walltime")
comm.Barrier()

glb_blk_wt = None
glb_blk_t = None
glb_blk_e0 = None
glb_blk_e1 = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_t = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_e0 = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_e1 = np.zeros(size * sampler.n_blocks,dtype="float32")
    ept_samples = np.zeros(sampler.n_blocks,dtype="float32")
comm.Barrier()
    
for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_t, blk_e0, blk_e1) =\
        sample_ccsd_pt.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_t = np.array([blk_t], dtype="float32")
    blk_e0 = np.array([blk_e0], dtype="float32")
    blk_e1 = np.array([blk_e1], dtype="float32")

    gather_wt = None
    gather_t = None
    gather_e0 = None
    gather_e1 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_t = np.zeros(size, dtype="float32")
        gather_e0 = np.zeros(size, dtype="float32")
        gather_e1 = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_t, gather_t, root=0)
    comm.Gather(blk_e0, gather_e0, root=0)
    comm.Gather(blk_e1, gather_e1, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_t[n * size : (n + 1) * size] = gather_t
        glb_blk_e0[n * size : (n + 1) * size] = gather_e0
        glb_blk_e1[n * size : (n + 1) * size] = gather_e1

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_t = np.sum(gather_wt * gather_t) / blk_wt
        blk_e0 = np.sum(gather_wt * gather_e0) / blk_wt
        blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_t, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)

    blk_ept = blk_e0 + blk_e1 - blk_t * (blk_e0 - h0)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_ept

    comm.Barrier()
    if rank == 0:
        ept_samples[n] = blk_ept
    comm.Barrier()

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:
            t = np.sum(glb_blk_wt * glb_blk_t)/np.sum(glb_blk_wt)
            e0 = np.sum(glb_blk_wt * glb_blk_e0)/np.sum(glb_blk_wt)
            e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

            ept = e0 + e1 - t * (e0 - h0)
            dE = np.array([-e0+h0,1-t,1])
            cov_te0e1 = np.cov([glb_blk_t[:(n+1)*size],
                                glb_blk_e0[:(n+1)*size],
                                glb_blk_e1[:(n+1)*size]])
            ept_err = np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt((n+1)*size)
            print(f"  {n:4d} \t \t {ept:.6f} \t {ept_err:.6f} \t"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()


comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None

    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_t,
                    glb_blk_e0,
                    glb_blk_e1,
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_t = samples_clean[:, 1]
    glb_blk_e0 = samples_clean[:, 2]
    glb_blk_e1 = samples_clean[:, 3]

    t = np.sum(glb_blk_wt * glb_blk_t)/np.sum(glb_blk_wt)
    e0 = np.sum(glb_blk_wt * glb_blk_e0)/np.sum(glb_blk_wt)
    e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

    ept = e0 + e1 - t*(e0-h0)
    dE = np.array([-e0+h0,1-t,1])
    cov_te0e1 = np.cov([glb_blk_t,glb_blk_e0,glb_blk_e1])
    ept_err = np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt(samples_clean.shape[0])

    ept_err = f"{ept_err:.6f}"

    ept = f"{ept:.6f}"

    print(f"Final Results1:")
    print(f"AFQMC/CCSD_PT energy: {ept} +/- {ept_err}")

    d = np.abs(ept_samples-np.median(ept_samples))
    d_med = np.median(d) + 1e-7
    mask = d/d_med < 10
    ept_clean = ept_samples[mask]
    print('# remove outliers: ', len(ept_samples)-len(ept_clean))

    ept_mean = np.mean(ept_clean)
    ept_std = np.std(ept_clean)

    ept = f"{ept_mean:.6f}"
    ept_err = f"{ept_std/np.sqrt(n):.6f}"

    print(f"Final Results2:")
    print(f"AFQMC/CCSD_PT energy: {ept} +/- {ept_err}")
    print(f"total run time: {time.time() - init_time:.2f}")

comm.Barrier()
